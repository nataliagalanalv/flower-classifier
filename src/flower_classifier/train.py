"""
Unified training script for every model in the project.

Pick the model with --model. Its defaults reproduce the experiments documented
in docs/results.md, and any of them can be overridden from the command line.

Examples:
    python -m flower_classifier.train --model cnn
    python -m flower_classifier.train --model resnet18-ft --epochs 10
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .data_setup import get_dataloaders
from .engine import evaluate, train_one_epoch
from .model import SimpleCNN
from .model_transfer import build_resnet18_feature_extractor, build_resnet18_finetune


@dataclass(frozen=True)
class ModelConfig:
    """Everything that differs between the models; the training loop is shared."""
    build_model: Callable      # builds the (untrained) model
    trainable_params: Callable  # (model, lr) -> parameters/groups handed to the optimizer
    img_size: int              # input size the model expects
    epochs: int
    lr: float
    weight_decay: float
    best_ckpt: str             # checkpoint with the best validation accuracy
    final_ckpt: str            # checkpoint of the last epoch


def _all_params(model, lr):
    return model.parameters()


def _head_params(model, lr):
    # Backbone is frozen: only the new classification head is optimized.
    return model.fc.parameters()


def _finetune_params(model, lr):
    # layer4 is pretrained -> gentle lr (1/10 of the head's).
    # fc is new (random init) -> full lr.
    return [
        {"params": model.layer4.parameters(), "lr": lr * 0.1},
        {"params": model.fc.parameters(), "lr": lr},
    ]


MODELS = {
    "cnn": ModelConfig(
        build_model=SimpleCNN, trainable_params=_all_params,
        img_size=128, epochs=30, lr=1e-3, weight_decay=1e-4,
        best_ckpt="best_simple_cnn_flowers.pth", final_ckpt="simple_cnn_flowers.pth",
    ),
    "resnet18-fe": ModelConfig(
        build_model=build_resnet18_feature_extractor, trainable_params=_head_params,
        img_size=224, epochs=20, lr=1e-3, weight_decay=0.0,
        best_ckpt="best_resnet18_feature_extract.pth", final_ckpt="resnet18_feature_extract_final.pth",
    ),
    "resnet18-ft": ModelConfig(
        build_model=build_resnet18_finetune, trainable_params=_finetune_params,
        img_size=224, epochs=20, lr=1e-3, weight_decay=0.0,
        best_ckpt="best_resnet18_finetune.pth", final_ckpt="resnet18_finetune_final_wd.pth",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a flower classifier.")
    parser.add_argument("--model", required=True, choices=list(MODELS), help="model to train")
    parser.add_argument("--epochs", type=int, default=None, help="override the model's default epochs")
    parser.add_argument("--lr", type=float, default=None, help="override the model's default learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="override the model's default weight decay")
    parser.add_argument("--output-dir", default="outputs", help="where checkpoints are saved")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MODELS[args.model]

    # `is not None` (not `or`): 0.0 is a valid weight decay and must not fall back to the default.
    epochs = args.epochs if args.epochs is not None else cfg.epochs
    lr = args.lr if args.lr is not None else cfg.lr
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg.weight_decay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    loaders = get_dataloaders(img_size=cfg.img_size)
    model = cfg.build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        cfg.trainable_params(model, lr), lr=lr, weight_decay=weight_decay
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0  # tracked across epochs, initialized once before the loop
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

        print(f"Época {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save whenever this epoch beats the best validation accuracy so far.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / cfg.best_ckpt)
            print(f"  → Nuevo mejor modelo guardado (Val Acc: {val_acc:.4f})")

    torch.save(model.state_dict(), output_dir / cfg.final_ckpt)
    print(f"Modelo final guardado en {output_dir / cfg.final_ckpt}")


if __name__ == "__main__":
    main()