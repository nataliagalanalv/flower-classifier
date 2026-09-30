"""
Fine-tuning script for ResNet18: backbone frozen except for `layer4` + `fc`.

Uses two parameter groups with different learning rates — a low lr for the unfrozen pretrained layer4 
(avoid destroying its ImageNet-learned features), a higher lr for the new fc head 
(random init, nothing to preserve there).
This run does NOT use weight_decay (see train_finetune_wd.py for that comparison) — 
this was the best-performing configuration overall (~90.78% val acc).
"""

import torch
import torch.nn as nn
from data_setup_tl import train_loader_tl, val_loader_tl
from model_transfer import build_resnet18_finetune

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model = build_resnet18_finetune(num_classes=102).to(device)
criterion = nn.CrossEntropyLoss()

# Two parameter groups, each with its own learning rate:
# - layer4: pretrained, unfrozen -> small lr, adjust gently.
# - fc: newly created, random weights -> larger lr, can learn faster.
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 0.0001},
    {"params": model.fc.parameters(), "lr": 0.001}
])

NUM_EPOCHS = 20
best_val_acc = 0.0 # tracked across epochs, initialized once before the loop

for epoch in range(NUM_EPOCHS):
    model.train() # enables Dropout / batch-based BatchNorm stats
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader_tl:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # clear gradients from the previous batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # backprop through layer4 + fc only (rest is frozen)
        optimizer.step() # each param group updated with its own lr

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

    # Checkpointing: save whenever this epoch beats the best validation accuracy seen so far — 
    # this is the file evaluate.py loads for the final test-set evaluation.
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "outputs/best_resnet18_finetune.pth")
        print(f"  → Nuevo mejor modelo guardado (Val Acc: {val_acc:.4f})")

# Keep the final-epoch state separately, for comparison against the best checkpoint.
torch.save(model.state_dict(), "outputs/resnet18_finetune_final_wd.pth")
print("Modelo final guardado en outputs/resnet18_finetune_final_wd.pth")