import copy

import pytest
import torch

from flower_classifier.engine import evaluate, train_one_epoch
from test_engine import make_toy_setup

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")

CPU = torch.device("cpu")
GPU = torch.device("cuda")


def test_evaluate_gives_same_result_on_cpu_and_gpu():
    model, loader, criterion, _ = make_toy_setup()
    loss_cpu, acc_cpu = evaluate(model, loader, criterion, CPU)
    loss_gpu, acc_gpu = evaluate(copy.deepcopy(model).to(GPU), loader, criterion, GPU)
    assert float(loss_gpu) == pytest.approx(float(loss_cpu), rel=1e-4)
    assert float(acc_gpu) == pytest.approx(float(acc_cpu))


def test_train_one_epoch_runs_on_gpu():
    model, loader, criterion, _ = make_toy_setup()
    model = model.to(GPU)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # created after moving the model
    loss, acc = train_one_epoch(model, loader, criterion, optimizer, GPU)
    assert all(p.is_cuda for p in model.parameters())
    assert loss >= 0
    assert 0.0 <= acc <= 1.0