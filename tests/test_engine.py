import pytest
import torch
import math
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flower_classifier.engine import evaluate, train_one_epoch

DEVICE = torch.device("cpu")

class GradModeSpy(nn.Module):
    """Linear model that records whether autograd was enabled during each forward pass."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.grad_enabled_calls = []

    def forward(self, x):
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        return self.linear(x)

def make_constant_model():
    """Linear model with zero weights and bias [2, 0]: always outputs logits [2, 0], so it always predicts class 0."""
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.copy_(torch.tensor([2.0, 0.0]))
    return model

def make_exact_case():
    """5 samples in batches of 2, 2 and 1. The label is 1 only once, so the model is right 4 times out of 5."""
    x = torch.zeros(5, 2)
    y = torch.tensor([0, 0, 0, 1, 0])
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
    loss_if_label_0 = math.log(1 + math.exp(-2.0))  # -log(softmax) of the predicted class
    loss_if_label_1 = math.log(1 + math.exp(2.0))   # -log(softmax) of the other class
    expected_loss = (4 * loss_if_label_0 + loss_if_label_1) / 5
    return loader, expected_loss, 0.8
    
def make_toy_setup():
    """Tiny, linearly separable problem: the class is the sign of the first feature."""
    torch.manual_seed(0)
    x = torch.tensor([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]] * 8)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
    model = nn.Linear(2, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, loader, criterion, optimizer


def test_train_one_epoch_updates_weights():
    model, loader, criterion, optimizer = make_toy_setup()
    before = [p.clone() for p in model.parameters()]
    train_one_epoch(model, loader, criterion, optimizer, DEVICE)
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


def test_evaluate_does_not_update_weights():
    model, loader, criterion, _ = make_toy_setup()
    before = [p.clone() for p in model.parameters()]
    evaluate(model, loader, criterion, DEVICE)
    assert all(torch.equal(b, p) for b, p in zip(before, model.parameters()))


def test_metrics_are_in_valid_range():
    model, loader, criterion, optimizer = make_toy_setup()
    loss, acc = train_one_epoch(model, loader, criterion, optimizer, DEVICE)
    assert loss >= 0
    assert 0.0 <= acc <= 1.0


def test_train_one_epoch_sets_train_mode():
    model, loader, criterion, optimizer = make_toy_setup()
    model.eval()
    train_one_epoch(model, loader, criterion, optimizer, DEVICE)
    assert model.training


def test_evaluate_sets_eval_mode():
    model, loader, criterion, _ = make_toy_setup()
    model.train()
    evaluate(model, loader, criterion, DEVICE)
    assert not model.training


def test_model_can_overfit_toy_data():
    model, loader, criterion, optimizer = make_toy_setup()
    loss_start, _ = evaluate(model, loader, criterion, DEVICE)
    for _ in range(30):
        train_one_epoch(model, loader, criterion, optimizer, DEVICE)
    loss_end, acc_end = evaluate(model, loader, criterion, DEVICE)
    assert loss_end < loss_start
    assert acc_end == 1.0

def test_evaluate_runs_without_autograd():
    _, loader, criterion, _ = make_toy_setup()
    model = GradModeSpy()
    evaluate(model, loader, criterion, DEVICE)
    assert model.grad_enabled_calls and not any(model.grad_enabled_calls)

def test_evaluate_returns_exact_loss_and_accuracy():
    loader, expected_loss, expected_acc = make_exact_case()
    loss, acc = evaluate(make_constant_model(), loader, nn.CrossEntropyLoss(), DEVICE)
    assert float(loss) == pytest.approx(expected_loss)
    assert float(acc) == pytest.approx(expected_acc)


def test_train_one_epoch_returns_exact_loss_and_accuracy():
    loader, expected_loss, expected_acc = make_exact_case()
    model = make_constant_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: the model never changes
    loss, acc = train_one_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, DEVICE)
    assert float(loss) == pytest.approx(expected_loss)
    assert float(acc) == pytest.approx(expected_acc)

