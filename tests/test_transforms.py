import numpy as np
import pytest
import torch
from PIL import Image

from flower_classifier.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
)


def make_noise_image(size=(300, 200)):
    """Random RGB image (width, height), so crops and color jitter produce visible differences."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


@pytest.mark.parametrize("img_size", [128, 224])
def test_eval_transform_outputs_square_tensor(img_size):
    out = build_eval_transform(img_size)(make_noise_image((300, 200)))
    assert out.shape == (3, img_size, img_size)


@pytest.mark.parametrize("img_size", [128, 224])
def test_train_transform_outputs_square_tensor(img_size):
    out = build_train_transform(img_size)(make_noise_image((300, 200)))
    assert out.shape == (3, img_size, img_size)


def test_eval_transform_is_deterministic():
    transform = build_eval_transform(128)
    image = make_noise_image()
    assert torch.equal(transform(image), transform(image))


def test_train_transform_is_random():
    transform = build_train_transform(128)
    image = make_noise_image()
    outputs = [transform(image) for _ in range(10)]
    assert any(not torch.equal(outputs[0], o) for o in outputs[1:])


@pytest.mark.parametrize("color", [0, 255])
def test_eval_transform_applies_imagenet_normalization(color):
    out = build_eval_transform(64)(Image.new("RGB", (100, 80), (color, color, color)))
    for channel in range(3):
        expected = (color / 255 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert torch.allclose(out[channel], torch.full_like(out[channel], expected), atol=1e-4)

@pytest.mark.parametrize(
    "color, expected",
    [
        (255, [2.2489, 2.4286, 2.6400]),
        (0, [-2.1179, -2.0357, -1.8044]),
    ],
)
def test_eval_transform_matches_known_imagenet_values(color, expected):
    out = build_eval_transform(64)(Image.new("RGB", (100, 80), (color, color, color)))
    for channel in range(3):
        target = torch.full_like(out[channel], expected[channel])
        assert torch.allclose(out[channel], target, atol=1e-3)