# Flower Classifier (Oxford 102 Flowers)

Image classifier for the [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset, built with PyTorch. Compares a convolutional neural network trained from scratch against transfer learning with ResNet18 (feature extraction and fine-tuning).

**Best result: 87.87% accuracy on the held-out test set**, using a fine-tuned ResNet18 (last residual block unfrozen). See [`RESULTS.md`](./RESULTS.md) for the full experiment log and [`ANALYSIS.md`](./ANALYSIS.md) for the write-up of findings and conclusions.

## Project structure

```
flower-classifier/
├── src/
│   ├── explore_data.py            # quick, raw dataset exploration
│   ├── data_setup.py              # data pipeline for the from-scratch CNN (128x128)
│   ├── data_setup_tl.py           # data pipeline for transfer learning (224x224, incl. test split)
│   ├── model.py                   # SimpleCNN architecture (from scratch)
│   ├── model_transfer.py          # ResNet18 builders (feature extraction / fine-tuning)
│   ├── train.py                   # trains SimpleCNN
│   ├── train_feature_extract.py   # trains ResNet18 with the backbone frozen
│   ├── train_finetune.py          # trains ResNet18 with layer4 unfrozen
│   ├── evaluate.py                # final, one-time evaluation on the test split
│   └── confusion_analysis.py      # confusion matrix, per-class accuracy, hardest classes
├── data/                          # dataset (downloaded automatically, gitignored)
├── outputs/                       # model checkpoints (.pth) and confusion_matrix.png
├── RESULTS.md                     # full results log (in Spanish)
├── ANALYSIS.md                    # findings and conclusions (in Spanish)
└── requirements.txt
```

## Setup

Tested with Python 3.14 on Windows 11 with an NVIDIA GPU (CUDA 12.6).

1. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```

2. Install PyTorch with CUDA support **first**, using the official selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) — the exact command depends on your GPU driver's supported CUDA version. Example:
   ```powershell
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
   > `requirements.txt` intentionally does not pin `torch`/`torchvision`, since a plain `pip install -r requirements.txt` would likely install a CPU-only build instead of the CUDA one. Install them separately first, then proceed with the rest of the dependencies below. Verify with:
   > ```python
   > import torch
   > print(torch.__version__, torch.cuda.is_available())
   > ```

3. Install the remaining dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

Run scripts from the project root, in this order:

```powershell
# 1. (Optional) Explore the raw dataset
python src/explore_data.py

# 2. Train the from-scratch CNN
python src/train.py

# 3. Train ResNet18 (feature extraction)
python src/train_feature_extract.py

# 4. Train ResNet18 (fine-tuning, layer4 unfrozen)
python src/train_finetune.py

# 5. Evaluate the best fine-tuned model on the held-out test split
python src/evaluate.py

# 6. Error analysis: confusion matrix, per-class accuracy, hardest classes
python src/confusion_analysis.py
```

The dataset downloads automatically on first run (`download=True`) into `data/`.

## Approach summary

| Approach | Trainable params | Val Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| CNN from scratch | ~11.2M (100%) | 37.65% | — |
| ResNet18 feature extraction | 52,326 (0.47%) | 79.22% | — |
| **ResNet18 fine-tuning** (`layer4` + `fc`) | 8,446,054 (~75%) | 90.78% | **87.87%** |

Full breakdown, per-class accuracy, and the most frequent misclassifications are in [`RESULTS.md`](./RESULTS.md). Interpretation of *why* transfer learning wins here, the effect (or lack thereof) of `weight_decay`, and dataset-imbalance caveats are in [`ANALYSIS.md`](./ANALYSIS.md).

## Notes on data sources

- Class names (flower species) come from a community-maintained `cat_to_name.json` mapping, not an official torchvision/Oxford source — treat names as approximate; the numeric class index (0–101) is the reliable reference. See `ANALYSIS.md` for a known inconsistency found in this file.