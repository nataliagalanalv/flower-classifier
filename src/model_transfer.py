"""
Transfer learning model builders based on ResNet18 pretrained on ImageNet.

Two strategies are provided:
- build_resnet18_feature_extractor: freezes the entire backbone, trains only a new classification head. Fast, few trainable params, good baseline.
- build_resnet18_finetune: also unfreezes the last residual block (layer4), allowing the model to adapt its most task-specific features to flowers.

Both replace the final `fc` layer (originally 1000 ImageNet classes) with a new one sized for our 102 flower classes.
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_resnet18_feature_extractor(num_classes=102):
    """Build ResNet18 with the entire pretrained backbone frozen.

    Only the newly added `fc` layer is trainable (it's created after the freezing loop, so it keeps its default requires_grad=True).
    """
    weights = ResNet18_Weights.DEFAULT # current recommended API (replaces pretrained=True)
    model = resnet18(weights=weights)   # downloads ImageNet-pretrained weights

    for param in model.parameters():
        param.requires_grad = False # freeze everything first

    num_features = model.fc.in_features  # 512 for ResNet18, read dynamically
    model.fc = nn.Linear(num_features, num_classes) # new head, trainable by default

    return model

def build_resnet18_finetune(num_classes=102):
    """Build ResNet18 with the backbone frozen except for `layer4` + `fc`.

    `layer4` is the last residual block — closest to the output, 
    and the one capturing the most task-specific (vs. generic) visual features. 
    Unfreezing only this block lets the model adapt to flowers without retraining the
    whole network from its pretrained state.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False # start fully frozen, same as feature extraction

    for param in model.layer4.parameters():
        param.requires_grad = True # selectively unfreeze the last block

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


if __name__ == "__main__":
    import torch

     # Sanity check 1: confirm feature extraction only trains the new head.
    model = build_resnet18_feature_extractor(num_classes=102)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Feature Extraction] Parámetros entrenables: {trainable_params:,} de {total_params:,} totales")

    # Sanity check 2: confirm fine-tuning unfreezes layer4 + fc (expected to be a much larger fraction of total params than feature extraction).
    model_ft = build_resnet18_finetune(num_classes=102)

    trainable_ft = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    total_ft = sum(p.numel() for p in model_ft.parameters())
    print(f"[Fine-tuning] Parámetros entrenables: {trainable_ft:,} de {total_ft:,} totales")

    # Sanity check 3: confirm the model can process a dummy input and produce the expected output shape.
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Shape de salida: {output.shape}")