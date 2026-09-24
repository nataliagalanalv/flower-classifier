"""
Data pipeline for the from-scratch CNN (128x128 images).

Builds the training and validation DataLoaders for the Oxford 102 Flowers dataset, including data augmentation for training and ImageNet normalization stats (required later for compatibility with the transfer learning pipeline).
"""

from torchvision import transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader

IMG_SIZE = 128
BATCH_SIZE = 32

# Training transform: includes data augmentation to compensate for the small training set (1020 images / 102 classes), reducing the risk of overfitting.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),  # random crop + resize, varies framing
    transforms.RandomHorizontalFlip(),                          # mirrors the image; flowers stay valid flipped
    transforms.RandomRotation(degrees=20),                      # simulates camera angle variation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # simulates lighting variation
    transforms.ToTensor(),                                      # PIL image -> tensor, scales pixels to [0, 1]
    # ImageNet mean/std: standard normalization, kept consistent with the transfer learning pipeline even though this model is trained from scratch.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = Flowers102(
    root="data",
    split="train",
    transform=train_transform,
    download=True # no-op if the dataset is already present in root
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True  # shuffle each epoch so the model doesn't learn from data order
)

# Validation transform: same resize/normalization as training, but WITHOUT augmentation — validation must measure performance on unmodified images.
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = Flowers102(
    root="data",
    split="val",
    transform=val_transform,
    download=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False # order doesn't matter when only evaluating, not training
)

if __name__ == "__main__":
    # Quick sanity check: confirm batch shapes and the effect of normalization
    # (pixel values should no longer be in [0, 1] after Normalize).
    images, labels = next(iter(train_loader))
    print(f"Shape del batch de imágenes: {images.shape}")
    print(f"Shape del batch de etiquetas: {labels.shape}")
    print(f"Rango de valores en la imagen: [{images.min():.3f}, {images.max():.3f}]")

