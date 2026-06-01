# Tuberculosis Detection from Chest X-Rays using Vision Transformers

This repository contains a deep learning project for binary classification of chest X-ray images (Normal vs. Tuberculosis) using state-of-the-art vision models:
1. **Vision Transformer (ViT-B/16)** — Fully Fine-tuned
2. **Swin Transformer V2 Large (SwinV2-L)** — Hierarchical Vision Transformer

---

## Dataset Overview
The dataset contains chest X-ray images classified into two categories:
* **Normal**: 514 images
* **Tuberculosis (TB)**: 2,494 images

### Data Splitting Strategy
The dataset is split into Train, Validation, and Test sets using a 70:15:15 ratio:
* **Train**: 2,104 images (Normal: 359, Tuberculosis: 1,745)
* **Validation**: 451 images (Normal: 77, Tuberculosis: 374)
* **Test**: 453 images (Normal: 78, Tuberculosis: 375)

*Note: Since the dataset exhibits significant class imbalance, a PyTorch `WeightedRandomSampler` is integrated into the training DataLoader to ensure balanced batch distributions.*

---

## Environment Configuration

This project is configured to run inside a Python environment configured for deep learning.

### 1. Install Dependencies
You can install all required dependencies by running the following command:
```bash
pip install -r requirements.txt
```

### 2. Core Dependencies
The primary libraries used in this implementation include:
* `torch` and `torchvision` (Deep learning framework)
* `timm` (PyTorch Image Models registry for SwinV2)
* `scikit-learn` (Metrics: classification report, Cohen's Kappa, confusion matrix)
* `matplotlib` and `seaborn` (Data visualization)
* `tqdm` (Progress tracking)

---

## Project Execution
To execute the project, run the cells in `chestvit6class.ipynb` sequentially:
1. **Data Preparation**: Extracts and structures the raw dataset into PyTorch `ImageFolder` compatible directories.
2. **Data Exploration**: Performs verification checks on split counts, evaluates imbalance ratios, and visualizes class distributions.
3. **ViT-B/16 Model Training**: Loads the ViT architecture with default pretrained weights, updates the head classifier for 2 classes, and conducts full fine-tuning.
4. **SwinV2 Large Model Training**: Pulls `swinv2_large` from the `timm` library, configures image resizing transforms, and trains using AdamW and CosineAnnealingLR schedulers.

---

## Model Evaluation
Models are evaluated on the held-out test set using standard machine learning metrics:
* Accuracy and Loss curves per epoch
* Weighted Precision, Recall, and F1-Score
* Cohen's Kappa Coefficient
* Confusion Matrices
