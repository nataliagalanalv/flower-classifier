"""Classify a single image and print the top-k flower predictions."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from .class_names import idx_to_name
from .model_transfer import RESNET_INPUT_SIZE, load_finetuned_model
from .transforms import build_eval_transform

DEFAULT_CHECKPOINT = "outputs/best_resnet18_finetune.pth"

@torch.no_grad()
def predict_top_k(model, image, device, k=5):
    """Return the k most likely classes for a PIL image as [(name, probability), ...]."""
    image = image.convert("RGB")  # PNGs may have 4 channels, some JPEGs are grayscale
    transform = build_eval_transform(RESNET_INPUT_SIZE)
    batch = transform(image).unsqueeze(0).to(device)  # [3, H, W] -> [1, 3, H, W]
    probs = torch.softmax(model(batch), dim=1)[0]  # 102 logits -> probabilities
    top_probs, top_idx = torch.topk(probs, k)
    return [(idx_to_name(i), p) for i, p in zip(top_idx.tolist(), top_probs.tolist())]

def main():
    parser = argparse.ArgumentParser(description="Classify a flower image.")
    parser.add_argument("image", help="path to the image file")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        parser.error(f"image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_finetuned_model(args.checkpoint, device)

    with Image.open(image_path) as image:
        results = predict_top_k(model, image, device, k=args.top_k)

    for name, prob in results:
        print(f"{name:<25} {prob:.1%}")


if __name__ == "__main__":
    main()