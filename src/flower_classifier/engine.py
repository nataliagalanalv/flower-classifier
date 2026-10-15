"""
Shared training and evaluation loops.

Every training script used to carry its own copy of these loops. They live
here once, so a fix or improvement applies to every model.
"""

import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one full pass over `loader`, updating the model's weights.
        Returns (average_loss, accuracy) over the whole epoch.
    """
    model.train()  # enables Dropout / batch-based BatchNorm stats
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # clear gradients from the previous batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # backprop: computes gradients via autograd
        optimizer.step()  # updates only the parameters given to the optimizer

        # Weighted by batch size: CrossEntropyLoss returns a per-batch mean,
        # which matters for the last, possibly smaller, batch.
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)  # index of the highest logit = predicted class
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()  # no backward() here, so no need to build the autograd graph
def evaluate(model, loader, criterion, device):
    """Measure loss and accuracy over `loader` without training.

    Returns (average_loss, accuracy).
    """
    model.eval()  # disables Dropout, uses running BatchNorm stats
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total