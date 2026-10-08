"""
DataLoader factory for the Oxford 102 Flowers dataset.

Nothing is built at import time: call get_dataloaders() when the data is needed.
"""

from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102

from .transforms import build_eval_transform, build_train_transform

BATCH_SIZE = 32


def get_dataloaders(img_size, batch_size=BATCH_SIZE, include_test=False, root="data"):
    """Return a dict of DataLoaders: "train", "val" and, only if requested, "test".

    The test split is excluded by default so training code cannot touch it
    by accident; it must be requested explicitly (see evaluate.py).
    """
    train_dataset = Flowers102(
        root=root, split="train",
        transform=build_train_transform(img_size), download=True,
    )
    val_dataset = Flowers102(
        root=root, split="val",
        transform=build_eval_transform(img_size), download=True,
    )

    loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    }

    if include_test:
        test_dataset = Flowers102(
            root=root, split="test",
            transform=build_eval_transform(img_size), download=True,
        )
        loaders["test"] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return loaders


def main():
    """Quick sanity check: confirm batch shapes and the effect of normalization."""
    loaders = get_dataloaders(img_size=224)
    images, labels = next(iter(loaders["train"]))
    print(f"Shape del batch de imágenes: {images.shape}")
    print(f"Shape del batch de etiquetas: {labels.shape}")
    print(f"Rango de valores en la imagen: [{images.min():.3f}, {images.max():.3f}]")


if __name__ == "__main__":
    main()