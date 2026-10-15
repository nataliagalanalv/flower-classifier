"""
Final, one-time evaluation of the best fine-tuned model on the test split.

This is the unbiased performance estimate for the project: unlike the validation split (used repeatedly during development to compare configurations), 
the test split has never influenced any decision made so far. The resulting numbers should be treated as the final, 
reportable result — not a starting point for further tuning.
"""

import argparse

import torch
import torch.nn as nn

from .data_setup import get_dataloaders
from .engine import evaluate
from .model_transfer import RESNET_INPUT_SIZE, load_finetuned_model

DEFAULT_CHECKPOINT = "outputs/best_resnet18_finetune.pth"


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned model on the test split.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="checkpoint to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # The test split must be requested explicitly (see get_dataloaders).
    test_loader = get_dataloaders(img_size=RESNET_INPUT_SIZE, include_test=True)["test"]
    model = load_finetuned_model(args.checkpoint, device)

    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()