"""
Quick, throwaway exploration of the raw Oxford 102 Flowers dataset.

No transforms applied here on purpose: the goal is to inspect the data as torchvision loads it by default 
(PIL image, original size, integer label) before deciding on the preprocessing pipeline in data_setup.py.
"""
from torchvision.datasets import Flowers102

# download=True is a no-op if the dataset is already present under root.
train_dataset = Flowers102(
    root="data",
    split="train",
    download=True
)

print(f"Número de imágenes en train: {len(train_dataset)}")

# Dataset.__getitem__ returns a (image, label) tuple. With no transform set,
# image is a raw PIL.Image (not a tensor) at its original, 
# non-fixed size — this dataset's images vary in resolution, 
# which is exactly why a Resize/RandomResizedCrop step is mandatory in the real training pipeline.
image, label = train_dataset[0]
print(f"Tipo de la imagen: {type(image)}")
print(f"Tamaño de la imagen: {image.size}")
print(f"Etiqueta (número de clase): {label}")