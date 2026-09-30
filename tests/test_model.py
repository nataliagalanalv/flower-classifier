"""
Unit tests for the from-scratch CNN architecture (SimpleCNN).
"""

import torch

from flower_classifier.model import SimpleCNN


def test_output_shape_matches_num_classes():
    """A forward pass should return one logit per class, per sample."""
    model = SimpleCNN(num_classes=102)
    dummy_input = torch.randn(1, 3, 128, 128)

    output = model(dummy_input)

    assert output.shape == (1, 102)


def test_output_shape_with_different_batch_size():
    """The batch dimension of the output should match the input's."""
    model = SimpleCNN(num_classes=102)
    dummy_input = torch.randn(4, 3, 128, 128)

    output = model(dummy_input)

    assert output.shape == (4, 102)


def test_output_shape_with_different_num_classes():
    """num_classes should control the size of the output layer."""
    model = SimpleCNN(num_classes=10)
    dummy_input = torch.randn(1, 3, 128, 128)

    output = model(dummy_input)

    assert output.shape == (1, 10)