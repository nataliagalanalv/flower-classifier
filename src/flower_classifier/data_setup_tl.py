"""
Data pipeline for transfer learning with ResNet18 (224x224 images).

Mirrors data_setup.py, but at the input size and normalization ResNet18 expects (pretrained on ImageNet). 
Provides train/val/test DataLoaders — the test split is only ever used once, for the final unbiased evaluation.
""" 

from torchvision import transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader

IMG_SIZE_TL = 224 # required input size for ResNet18 (matches its ImageNet pretraining)
BATCH_SIZE = 32

# Training transform: same augmentation strategy as the from-scratch CNN.
train_transform_tl = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE_TL, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    # ImageNet mean/std — mandatory here: ResNet18's pretrained weights expect inputs normalized exactly this way, not just "some" normalization.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Val/test transform: no augmentation, only resize + normalize. 
# Reused for both splits since neither should ever see randomized inputs.
val_transform_tl = transforms.Compose([
    transforms.Resize((IMG_SIZE_TL, IMG_SIZE_TL)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset_tl = Flowers102(
    root="data",
    split="train",
    transform=train_transform_tl,
    download=True
)

val_dataset_tl = Flowers102(
    root="data",
    split="val",
    transform=val_transform_tl,
    download=True
)

train_loader_tl = DataLoader(
    train_dataset_tl,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader_tl = DataLoader(
    val_dataset_tl,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Test split: held out from every training/tuning decision made during the project. 
# Only evaluated once, at the very end, for an unbiased performance estimate (see evaluate.py).
test_dataset_tl = Flowers102(
    root="data",
    split="test",
    transform=val_transform_tl,
    download=True
)

test_loader_tl = DataLoader(
    test_dataset_tl,
    batch_size=BATCH_SIZE,
    shuffle=False
)

if __name__ == "__main__":
     # Quick sanity check: confirm batch shape matches ResNet18's expected input.
    images, labels = next(iter(train_loader_tl))
    print(f"Shape del batch de imágenes: {images.shape}")
    print(f"Shape del batch de etiquetas: {labels.shape}")