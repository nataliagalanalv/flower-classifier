"""
Unit tests for the ResNet18-based transfer learning model builders.

Note: these tests call resnet18(weights=...), which downloads ImageNet
pretrained weights on first run (cached afterwards by torchvision). They
require network access the first time they execute.
"""

import torch

from flower_classifier.model_transfer import build_resnet18_feature_extractor, build_resnet18_finetune


def test_feature_extractor_only_trains_fc_layer():
    """Feature extraction should freeze everything except the new fc head."""
    model = build_resnet18_feature_extractor(num_classes=102)

    for name, param in model.named_parameters():
        if name.startswith("fc."):
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_feature_extractor_trainable_param_count():
    """52,326 trainable params matches fc: 512 * 102 weights + 102 biases."""
    model = build_resnet18_feature_extractor(num_classes=102)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert trainable_params == 52_326


def test_finetune_unfreezes_layer4_and_fc_only():
    """Fine-tuning should leave layer1-3 frozen, unfreeze layer4 and fc."""
    model = build_resnet18_finetune(num_classes=102)

    for name, param in model.named_parameters():
        if name.startswith("layer4.") or name.startswith("fc."):
            assert param.requires_grad is True
        elif name.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.")):
            assert param.requires_grad is False


def test_output_shape_matches_num_classes():
    """Both builders should produce one logit per class, per sample."""
    model = build_resnet18_finetune(num_classes=102)
    dummy_input = torch.randn(1, 3, 224, 224)

    output = model(dummy_input)

    assert output.shape == (1, 102)