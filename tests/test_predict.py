import pytest
import torch
from PIL import Image
from torch import nn

from flower_classifier.class_names import idx_to_name
from flower_classifier.predict import predict_top_k

DEVICE = torch.device("cpu")

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")),
]

class FixedLogits(nn.Module):
    """Dummy model: ignores the input and always returns the same logits (class 7 > 3 > 50)."""

    def __init__(self):
        super().__init__()
        logits = torch.zeros(102)
        logits[7], logits[3], logits[50] = 5.0, 4.0, 3.0
        self.register_buffer("logits", logits)

    def forward(self, x):
        return self.logits.repeat(x.shape[0], 1)


def make_image(mode="RGB"):
    return Image.new(mode, (60, 40))


def test_top_predictions_are_ordered_by_probability():
    results = predict_top_k(FixedLogits(), make_image(), DEVICE, k=3)
    assert [name for name, _ in results] == [idx_to_name(7), idx_to_name(3), idx_to_name(50)]


@pytest.mark.parametrize("k", [1, 5])
def test_returns_k_results(k):
    assert len(predict_top_k(FixedLogits(), make_image(), DEVICE, k=k)) == k


def test_probabilities_are_valid_and_descending():
    probs = [p for _, p in predict_top_k(FixedLogits(), make_image(), DEVICE, k=5)]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0 + 1e-6


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_accepts_non_rgb_images(mode):
    assert len(predict_top_k(FixedLogits(), make_image(mode), DEVICE, k=5)) == 5


@pytest.mark.parametrize("device", DEVICES)
def test_predict_runs_on_device(device):
    device = torch.device(device)
    results = predict_top_k(FixedLogits().to(device), make_image(), device, k=3)
    assert results[0][0] == idx_to_name(7)