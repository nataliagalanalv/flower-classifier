"""
Final, one-time evaluation of the best fine-tuned model on the test split.

This is the unbiased performance estimate for the project: unlike the validation split (used repeatedly during development to compare configurations), 
the test split has never influenced any decision made so far. The resulting numbers should be treated as the final, 
reportable result — not a starting point for further tuning.
"""

import torch
import torch.nn as nn
from data_setup_tl import test_loader_tl
from model_transfer import build_resnet18_finetune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Rebuild the exact architecture used during fine-tuning (state_dict only stores weight values, not the model structure), 
# then load the checkpoint with the best validation accuracy.
model = build_resnet18_finetune(num_classes=102)
model.load_state_dict(torch.load("outputs/best_resnet18_finetune.pth", weights_only=True))
model.to(device)
model.eval()  # disables Dropout, uses running BatchNorm stats (not batch stats)

criterion = nn.CrossEntropyLoss()

test_loss = 0.0
test_correct = 0
test_total = 0

# Single pass over the test set. No backward(), no optimizer step, 
# no gradient tracking — this loop only measures, it never trains.
with torch.no_grad():
    for images, labels in test_loader_tl:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Weighted by actual batch size (images.size(0)) 
        # because CrossEntropyLoss returns the per-batch mean, 
        # not the sum — matters for the last batch, which may be smaller than BATCH_SIZE.
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_loss = test_loss / test_total
test_acc = test_correct / test_total

print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")