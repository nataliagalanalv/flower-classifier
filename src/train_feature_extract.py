"""
Training script for ResNet18 feature extraction (backbone fully frozen).

Only the new `fc` classification head (~52K params) is trained; 
the rest of ResNet18 keeps its ImageNet-pretrained weights unchanged. 
Saves both the best checkpoint (by validation accuracy) and the final-epoch checkpoint.
"""

import torch
import torch.nn as nn
from data_setup_tl import train_loader_tl, val_loader_tl
from model_transfer import build_resnet18_feature_extractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model = build_resnet18_feature_extractor(num_classes=102).to(device)
criterion = nn.CrossEntropyLoss()
# Only model.fc.parameters() are passed here — the frozen backbone has no
# gradients to update, so there's no need (and no benefit) to pass them all.
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

NUM_EPOCHS = 20
best_val_acc = 0.0 # tracked across epochs, initialized once before the loop

for epoch in range(NUM_EPOCHS):
    model.train()  # enables Dropout / batch-based BatchNorm stats
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader_tl:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # clear gradients from the previous batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # backprop: computes gradients via autograd
        optimizer.step() # updates only the trainable params (model.fc)

        # Weighted by actual batch size since CrossEntropyLoss returns a
        # per-batch mean, not a sum (matters for the last, possibly smaller batch)
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1) # index of the highest logit = predicted class
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()  # disables Dropout, uses running BatchNorm stats
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): # no backward() here, so no need to build the autograd graph
        for images, labels in val_loader_tl:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    print(f"Época {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Checkpointing: save whenever this epoch beats the best validation accuracy seen so far 
    # (not just the final epoch's result).
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "outputs/best_resnet18_feature_extract.pth")
        print(f"  → Nuevo mejor modelo guardado (Val Acc: {val_acc:.4f})")

# Also keep the final-epoch state separately, for comparison against the best checkpoint.
torch.save(model.state_dict(), "outputs/resnet18_feature_extract_final.pth")
print("Modelo final guardado en outputs/resnet18_feature_extract_final.pth")