"""
Training script for the from-scratch CNN (SimpleCNN), 128x128 images.

All ~11.2M parameters are trained from random initialization — 
no pretrained weights involved. Uses weight_decay + BatchNorm (in the model)
+ data augmentation (in data_setup.py) together to reduce overfitting,
given the small training set (1020 images / 102 classes).
"""

import torch
import torch.nn as nn
from data_setup import train_loader, val_loader
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model = SimpleCNN(num_classes=102).to(device)
criterion = nn.CrossEntropyLoss()
# weight_decay (L2 regularization): penalizes large weights, pushing the
# model toward "smoother" solutions less likely to memorize the training set.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

NUM_EPOCHS = 30
best_val_acc = 0.0 # tracked across epochs, initialized once before the loop

for epoch in range(NUM_EPOCHS):
    model.train() # enables Dropout / batch-based BatchNorm stats
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # clear gradients from the previous batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # backprop: computes gradients via autograd
        optimizer.step() # updates all model parameters (nothing frozen here)

        # Weighted by actual batch size since CrossEntropyLoss returns a
        # per-batch mean, not a sum (matters for the last, possibly smaller batch).
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1) # index of the highest logit = predicted class
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval() # disables Dropout, uses running BatchNorm stats
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): # no backward() here, so no need to build the autograd graph
        for images, labels in val_loader:
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

    # Checkpointing: save whenever this epoch beats the best validation
    # accuracy seen so far (not just whatever the final epoch happens to be).
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "outputs/best_simple_cnn_flowers.pth")
        print(f"  → Nuevo mejor modelo guardado (Val Acc: {val_acc:.4f})")

# Also keep the final-epoch state separately, for comparison against the best checkpoint.
torch.save(model.state_dict(), "outputs/simple_cnn_flowers.pth")
print("Modelo final (época 30) guardado en outputs/simple_cnn_flowers.pth")