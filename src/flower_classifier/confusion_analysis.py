"""
Error analysis for the final fine-tuned model, on the held-out test split.

Loads the best fine-tuned ResNet18 checkpoint, runs it once over the test set, and reports: the most frequent misclassifications, per-class accuracy
(highlighting the hardest classes), and a full confusion matrix heatmap.

Class names come from a community-maintained cat_to_name.json mapping (not an official torchvision/Oxford source) — 
treat names as approximate and cross-check against the numeric class index if precision matters.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from .class_names import idx_to_name
from .data_setup import get_dataloaders
from .model_transfer import RESNET_INPUT_SIZE, load_finetuned_model

DEFAULT_CHECKPOINT = "outputs/best_resnet18_finetune.pth"

def collect_predictions(model, loader, device):
    """Run the model once over `loader` and return (true labels, predictions)."""
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _, predicted = torch.max(model(images), 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())  # already on CPU, never moved to device
    return all_labels, all_preds

def main():
    parser = argparse.ArgumentParser(description="Error analysis on the test split.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_finetuned_model(args.checkpoint, device)
    test_loader = get_dataloaders(img_size=RESNET_INPUT_SIZE, include_test=True)["test"]
    all_labels, all_preds = collect_predictions(model, test_loader, device)

    # Full confusion matrix: cm[true_class, pred_class] = count.
    cm = confusion_matrix(all_labels, all_preds)

    # Separate copy with the diagonal zeroed out, used only for ranking mistakes —
    # correct predictions shouldn't compete with actual misclassifications here.
    cm_for_ranking = cm.copy()
    np.fill_diagonal(cm_for_ranking, 0)

    confusions = []
    for true_class in range(cm_for_ranking.shape[0]):
        for pred_class in range(cm_for_ranking.shape[1]):
            count = cm_for_ranking[true_class, pred_class]
            if count > 0:
                confusions.append((true_class, pred_class, count))

    confusions.sort(key=lambda x: x[2], reverse=True)  # most frequent mistakes first

    print("Top 10 confusiones más frecuentes (clase real -> clase predicha : nº de veces):")
    for true_class, pred_class, count in confusions[:10]:
        true_name = idx_to_name(true_class)
        pred_name = idx_to_name(pred_class)
        print(f"  {true_name} (clase {true_class}) -> {pred_name} (clase {pred_class}) : {count} veces")

    # Per-class accuracy = diagonal (correct) / row sum (total true samples of that class).
    per_class_total = cm.sum(axis=1)  # support per class (row = true class)
    per_class_correct = np.diag(cm)

    # Guarded division: avoids a ZeroDivisionError/NaN if any class had 0 test samples (not expected with this dataset, but kept defensive).
    per_class_acc = np.divide(
        per_class_correct,
        per_class_total,
        out=np.zeros_like(per_class_correct, dtype=float),
        where=per_class_total != 0
    )

    class_stats = []
    for idx in range(len(per_class_acc)):
        class_stats.append((idx, idx_to_name(idx), per_class_acc[idx], per_class_total[idx]))

    class_stats.sort(key=lambda x: x[2]) # lowest accuracy first

    print("\nLas 15 clases más difíciles (menor accuracy en test):")
    for idx, name, acc, support in class_stats[:15]:
        print(f"  {name} (clase {idx}) : {acc:.2%} de acierto ({support} imágenes en test)")

    # NOTE: this is the unweighted mean across 102 per-class accuracies — 
    # different from the overall (sample-weighted) test accuracy computed in evaluate.py.
    # A large, poorly-classified class can pull the two metrics in opposite directions.
    overall_mean_per_class_acc = per_class_acc.mean()
    print(f"\nAccuracy media por clase (promedio no ponderado de las 102): {overall_mean_per_class_acc:.4f}")

    # Full 102x102 heatmap. Individual class labels aren't legible at this scale —
    # this is for spotting the general error pattern, not reading exact values;
    # use the per-class breakdown above for that.
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, cmap="viridis", cbar=True)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de confusión completa (102 clases) - Test set")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    print("Matriz de confusión guardada en outputs/confusion_matrix.png")

if __name__ == "__main__":
    main()