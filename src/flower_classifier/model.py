"""
From-scratch CNN architecture for 102-class flower classification.

4 convolutional blocks (32->64->128->256 filters) doubling channels while halving spatial size via MaxPool, 
followed by a classification head with Dropout for regularization. 
Designed for 128x128 inputs (see data_setup.py).
"""

import torch.nn as nn

class SimpleCNN(nn.Module):
    """Basic CNN: 4 conv blocks -> flatten -> dropout -> linear classifier.

    Architecture choices:
    - 3x3 kernels throughout (VGG-style: same receptive field as larger kernels, fewer parameters, matches ResNet18's convention too).
    - kernel_size=3 + padding=1 keeps spatial size unchanged after each conv; only MaxPool(2, 2) halves it, making the size progression predictable
      (128 -> 64 -> 32 -> 16 -> 8 across the 4 blocks).
    - BatchNorm after each conv: stabilizes/speeds up training, mild regularization side effect.
    - Filter count doubles per block (32->256): few filters early (simple patterns: edges, color), more filters late (complex patterns).
    """
    def __init__(self, num_classes=102):
        super().__init__()  # required when subclassing nn.Module

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5) # regularization; matters with only ~1k training images
        # in_features must match the flattened size after 4 blocks:
        # 256 channels x 8x8 spatial (128 / 2^4 = 8) = 16384.
        self.fc = nn.Linear(in_features=256 * 8 * 8, out_features=num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Sanity check: confirm the architecture's dimensions are consistent
    # (a mismatch in fc's in_features would raise a shape error here).
    import torch
    model = SimpleCNN(num_classes=102)
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)
    print(f"Shape de salida: {output.shape}")