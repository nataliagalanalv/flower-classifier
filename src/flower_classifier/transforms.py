"""
Image transforms shared by training, evaluation and inference.

Kept separate from the dataset/loader code so that importing a transform
(e.g. from predict.py or the API) never triggers a dataset download.
"""

from torchvision import transforms

# ImageNet statistics: ResNet18's pretrained weights expect inputs normalized
# exactly this way. The from-scratch CNN uses them too, for consistency.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(img_size):
    """Training pipeline: data augmentation + normalization."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(img_size):
    """Validation/test/inference pipeline: no randomness, only resize + normalize."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])