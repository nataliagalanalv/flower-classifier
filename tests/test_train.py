import sys

import pytest
import torch
from torch import nn

from flower_classifier import train
from flower_classifier.model_transfer import RESNET_INPUT_SIZE, build_resnet18_finetune
from flower_classifier.train import MODELS, ModelConfig, parse_args

# Tiny stand-in for a real model config, so main() can run in milliseconds.
TOY = ModelConfig(
    build_model=lambda: nn.Linear(2, 2),
    trainable_params=lambda model, lr: model.parameters(),
    img_size=8, epochs=3, lr=0.1, weight_decay=1e-4,
    best_ckpt="best.pth", final_ckpt="final.pth",
)


def set_argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["train", *args])


# --- model registry ---------------------------------------------------------

def test_registry_contains_the_three_models():
    assert set(MODELS) == {"cnn", "resnet18-fe", "resnet18-ft"}


def test_image_size_matches_each_model():
    assert MODELS["cnn"].img_size == 128
    assert MODELS["resnet18-fe"].img_size == RESNET_INPUT_SIZE
    assert MODELS["resnet18-ft"].img_size == RESNET_INPUT_SIZE


def test_checkpoint_names_are_unique():
    names = [name for cfg in MODELS.values() for name in (cfg.best_ckpt, cfg.final_ckpt)]
    assert len(names) == len(set(names))
    assert all(name.endswith(".pth") for name in names)


def test_finetune_uses_a_lower_lr_for_layer4_than_for_the_head():
    model = build_resnet18_finetune(pretrained=False)
    layer4_group, fc_group = MODELS["resnet18-ft"].trainable_params(model, 1e-3)
    assert layer4_group["lr"] == pytest.approx(1e-4)
    assert fc_group["lr"] == pytest.approx(1e-3)
    assert {id(p) for p in layer4_group["params"]} == {id(p) for p in model.layer4.parameters()}
    assert {id(p) for p in fc_group["params"]} == {id(p) for p in model.fc.parameters()}


# --- command-line arguments -------------------------------------------------

def test_model_argument_is_required(monkeypatch):
    set_argv(monkeypatch)
    with pytest.raises(SystemExit):
        parse_args()


def test_unknown_model_is_rejected(monkeypatch):
    set_argv(monkeypatch, "--model", "vgg")
    with pytest.raises(SystemExit):
        parse_args()


def test_optional_arguments_default_to_none(monkeypatch):
    set_argv(monkeypatch, "--model", "cnn")
    args = parse_args()
    assert args.epochs is None
    assert args.lr is None
    assert args.weight_decay is None
    assert args.output_dir == "outputs"


def test_arguments_are_parsed_with_the_right_types(monkeypatch):
    set_argv(monkeypatch, "--model", "cnn", "--epochs", "3", "--lr", "0.01", "--weight-decay", "0.0")
    args = parse_args()
    assert args.epochs == 3 and isinstance(args.epochs, int)
    assert args.lr == 0.01
    assert args.weight_decay == 0.0


# --- main(): wiring, with fake data and fake training steps -----------------

def run_main(monkeypatch, tmp_path, *cli_args, val_accs=(0.5, 0.6, 0.7)):
    """Run train.main() on the toy config and return what the fakes recorded."""
    record = {"epochs_run": 0, "img_size": None, "adam": {}}
    accs = iter(val_accs)

    def fake_get_dataloaders(img_size):
        record["img_size"] = img_size
        return {"train": None, "val": None}

    def fake_train_one_epoch(model, loader, criterion, optimizer, device):
        record["epochs_run"] += 1
        with torch.no_grad():
            model.weight.fill_(float(record["epochs_run"]))  # the weights reveal which epoch they come from
        return 0.0, 0.0

    def fake_evaluate(model, loader, criterion, device):
        return 0.0, next(accs)

    real_adam = torch.optim.Adam

    def spy_adam(params, **kwargs):
        record["adam"] = kwargs
        return real_adam(params, **kwargs)

    monkeypatch.setitem(MODELS, "toy", TOY)
    monkeypatch.setattr(train, "get_dataloaders", fake_get_dataloaders)
    monkeypatch.setattr(train, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train, "evaluate", fake_evaluate)
    monkeypatch.setattr(torch.optim, "Adam", spy_adam)
    set_argv(monkeypatch, "--model", "toy", "--output-dir", str(tmp_path / "out"), *cli_args)
    train.main()
    return record


def test_defaults_come_from_the_model_config(monkeypatch, tmp_path):
    record = run_main(monkeypatch, tmp_path)
    assert record["epochs_run"] == TOY.epochs
    assert record["img_size"] == TOY.img_size
    assert record["adam"] == {"lr": TOY.lr, "weight_decay": TOY.weight_decay}


def test_cli_overrides_replace_the_defaults(monkeypatch, tmp_path):
    record = run_main(monkeypatch, tmp_path, "--epochs", "2", "--lr", "0.5")
    assert record["epochs_run"] == 2
    assert record["adam"]["lr"] == 0.5


def test_zero_weight_decay_overrides_the_default(monkeypatch, tmp_path):
    record = run_main(monkeypatch, tmp_path, "--weight-decay", "0.0")
    assert record["adam"]["weight_decay"] == 0.0


def test_best_checkpoint_follows_validation_accuracy(monkeypatch, tmp_path):
    run_main(monkeypatch, tmp_path, val_accs=(0.5, 0.7, 0.6))
    best = torch.load(tmp_path / "out" / "best.pth", map_location="cpu", weights_only=True)
    final = torch.load(tmp_path / "out" / "final.pth", map_location="cpu", weights_only=True)
    assert torch.all(best["weight"] == 2.0)   # epoch 2 had the best validation accuracy
    assert torch.all(final["weight"] == 3.0)  # the final checkpoint is always the last epoch


def test_output_directory_is_created(monkeypatch, tmp_path):
    run_main(monkeypatch, tmp_path)
    assert (tmp_path / "out").is_dir()
    assert (tmp_path / "out" / "best.pth").is_file()
    assert (tmp_path / "out" / "final.pth").is_file()