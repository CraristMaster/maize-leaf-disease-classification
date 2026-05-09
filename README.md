# Maize Leaf Disease Classification using CNNs and Bayesian Hyperparameter Optimization

Comparative study of three CNN architectures (AlexNet, ResNet-50, SqueezeNet) for classifying maize leaf diseases from the PlantVillage dataset, using transfer learning, data augmentation, and Bayesian optimization for hyperparameter tuning. Validated with stratified 5-fold cross-validation. Published at the *XVI Workshop de Visão Computacional* (WVC 2020).

> Best result: **97% accuracy** across all three architectures, with F1-Score up to 96% (ResNet-50, SqueezeNet).

> **Note:** This repository contains **source code only**. The dataset and trained weights are not versioned here due to size.
>
> A complete mirror — including dataset, trained checkpoints, and additional experimental artifacts — is hosted on GitLab: **[larissafrodrigues/maize-leaf-disease-classification](https://gitlab.com/larissafrodrigues/maize-leaf-disease-classification)**.
>
> See [Dataset](#dataset) for download instructions and [Reproducing the Experiments](#reproducing-the-experiments) to retrain from scratch.

---

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Citation](#citation)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Maize is the most produced food crop in the world. Visual identification of leaf diseases is subjective, error-prone, and time-consuming, and inaccurate diagnoses lead to incorrect pesticide use, hurting yield and human health. This work investigates the use of CNNs as an automated, reproducible alternative.

We benchmark three pre-trained architectures on a four-class subset of PlantVillage and search for optimal training hyperparameters using Bayesian optimization with Gaussian Process surrogate. Compared to grid/random search, Bayesian optimization reaches comparable or better configurations in fewer evaluations, which matters under realistic compute budgets.

The full paper is available [here](https://sol.sbc.org.br/index.php/wvc/article/view/13489/13337).

## Highlights

- Performance comparison of three state-of-the-art CNNs (AlexNet, ResNet-50, SqueezeNet) for maize leaf disease classification.
- Hyperparameters (batch size, learning rate, momentum) tuned via Bayesian optimization.
- Stratified 5-fold cross-validation, more robust than the hold-out splits used by most prior work on this dataset.
- Data augmentation (random rotation, horizontal/vertical flips) to mitigate class imbalance.
- Fine-tuning from ImageNet weights on all three backbones.
- Best result: **97% accuracy**, comparable to or above prior literature on the same dataset.

## Dataset

Images from the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset), maize subset only.

| Class                | Samples |
| -------------------- | ------- |
| Gray Leaf Spot       | 513     |
| Common Rust          | 1192    |
| Northern Leaf Blight | 985     |
| Healthy              | 1162    |
| **Total**            | **3852**|

The dataset is imbalanced: Gray Leaf Spot is roughly 43% the size of Common Rust. This imbalance is partially addressed via data augmentation rather than resampling.

All images are resized to **224 × 224** to match the input shape expected by the pre-trained backbones.

### Getting the data

The dataset is **not included** in this repository. To set it up:

```bash
# Clone the PlantVillage dataset (large, only the maize folders are needed)
git clone https://github.com/spMohanty/PlantVillage-Dataset.git /tmp/plantvillage

# Copy only the maize classes into ./data/
mkdir -p data
cp -r /tmp/plantvillage/raw/color/Corn___Cercospora_leaf_spot\ Gray_leaf_spot data/gray_leaf_spot
cp -r /tmp/plantvillage/raw/color/Corn___Common_rust                          data/common_rust
cp -r /tmp/plantvillage/raw/color/Corn___Northern_Leaf_Blight                 data/northern_leaf_blight
cp -r /tmp/plantvillage/raw/color/Corn___healthy                              data/healthy
```

Expected layout after setup:

```
data/
├── gray_leaf_spot/        # 513 images
├── common_rust/           # 1192 images
├── northern_leaf_blight/  # 985 images
└── healthy/               # 1162 images
```

## Methodology

The pipeline has three stages:

1. **Data split** — the dataset is partitioned into 6 stratified folds. One fold is held out as the final test set; the remaining 5 are used for cross-validation.
2. **Hyperparameter search + training** — for each architecture, Bayesian optimization runs over the 5 cross-validation folds, scoring candidates by validation loss. The best model is then retrained.
3. **Test evaluation** — final metrics (accuracy, precision, recall, F1-score) are reported on the held-out test fold using macro-average.

### Training setup

- **Backbones**: AlexNet, ResNet-50, SqueezeNet — all initialized with ImageNet weights, then fine-tuned end-to-end (deeper layers adjusted).
- **Optimizer**: SGD with momentum.
- **Loss**: Cross-entropy.
- **Epochs**: 30.
- **Augmentation**: random rotation in `[0°, 360°]`, horizontal flip, vertical flip.
- **Validation**: stratified 5-fold cross-validation.

### Hyperparameter search space

Searched with Bayesian optimization (Gaussian Process surrogate) over a uniform prior:

| Hyperparameter | Range          |
| -------------- | -------------- |
| Batch size     | [16, 32]       |
| Learning rate  | [0.001, 0.01]  |
| Momentum       | [0, 1]         |

### Best hyperparameters found

| Hyperparameter | AlexNet  | ResNet-50 | SqueezeNet |
| -------------- | -------- | --------- | ---------- |
| Batch size     | 32       | 32        | 18         |
| Learning rate  | 0.003693 | 0.004250  | 0.002676   |
| Momentum       | 0.1387   | 0.4755    | 0.3456     |

## Results

Test performance per model (macro-average):

| Model      | Accuracy | Precision | Recall | F1-Score |
| ---------- | -------- | --------- | ------ | -------- |
| AlexNet    | 97%      | 96%       | 95%    | 95%      |
| ResNet-50  | 97%      | 95%       | 96%    | 96%      |
| SqueezeNet | 97%      | 95%       | 96%    | 96%      |

Per-class performance is detailed in the paper. Gray Leaf Spot, the smallest class, has the lowest scores — confusion matrices show moderate correlation with Common Rust and Healthy, suggesting visual overlap between these classes that augmentation alone does not fully resolve.

### Comparison with prior work on the same dataset

| Reference                       | Method                          | Accuracy |
| ------------------------------- | ------------------------------- | -------- |
| Sibiya & Sumbwanyambe (2019)    | CNN                             | 92.85%   |
| Hu et al. (2020)                | Pre-trained GoogLeNet           | 97.60%   |
| Priyadharshini et al. (2019)    | LeNet-5 variant + PCA whitening | 97.89%   |
| Bhatt et al. (2019)             | Inception-v2 + AdaBoost         | 98.00%   |
| **This work**                   | AlexNet / ResNet-50 / SqueezeNet + Bayesian opt. + 5-fold CV | **97.00%** |

Note: prior works rely on hold-out validation, while this study uses stratified k-fold cross-validation, which is more robust to overfitting and outliers. Reported accuracy is therefore not directly comparable.

## Requirements

- Python 3.6
- PyTorch 1.4
- CUDA 8.0, cuDNN 6.0
- `bayesian-optimization` 1.2.0
- NumPy, scikit-learn, matplotlib (for evaluation and plots)

### Hardware used in the experiments

- CPU: Intel Core i5 @ 3.00 GHz
- RAM: 16 GB
- GPU: NVIDIA GeForce GTX Titan Xp (12 GB)
- OS: Ubuntu 16.04.2 LTS

### Installation

```bash
git clone <this-repo>.git
cd <this-repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> The pinned versions reflect the original experimental environment. Newer PyTorch / CUDA versions should work for inference, but exact reproduction of the reported numbers is only guaranteed with the original stack.

## Project Structure

```
.
├── data/                  # PlantVillage maize subset — NOT versioned (see Dataset)
├── notebooks/             # EDA and result analysis
├── src/
│   ├── dataset.py         # Dataset loading, splits, augmentation
│   ├── models.py          # AlexNet / ResNet-50 / SqueezeNet wrappers
│   ├── train.py           # Training loop and CV orchestration
│   ├── optimize.py        # Bayesian optimization driver
│   └── evaluate.py        # Test metrics, confusion matrices
├── results/               # Per-fold metrics, best hyperparameters, plots
│   └── *.pt               # Trained checkpoints — NOT versioned
├── .gitignore
├── requirements.txt
└── README.md
```

> Adjust the tree above to match your actual layout. The repository ships **code only**; `data/` and trained model checkpoints (`.pt`, `.pth`) are excluded via `.gitignore`.

## Reproducing the Experiments

> Trained checkpoints are **not included** in this repository (file size). You need to retrain from scratch using the steps below. A typical full run (search + train for one architecture) takes a few hours on a Titan Xp-class GPU.

1. **Get the data.** Follow the instructions in [Dataset → Getting the data](#getting-the-data).
2. **Run hyperparameter search:**
   ```bash
   python src/optimize.py --model resnet50 --folds 5 --iterations 25
   ```
   This produces a JSON with the best hyperparameters under `results/<model>/best_params.json`.
3. **Train final model with best hyperparameters:**
   ```bash
   python src/train.py --model resnet50 --params results/resnet50/best_params.json --epochs 30
   ```
4. **Evaluate on the held-out test fold:**
   ```bash
   python src/evaluate.py --model resnet50 --checkpoint results/resnet50/best.pt
   ```

> Replace `resnet50` with `alexnet` or `squeezenet` to reproduce the other two backbones.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{Rocha2020,
  author    = {Erik Lucas da Rocha and Larissa Rodrigues and João Fernando Mari},
  title     = {Maize leaf disease classification using convolutional neural networks and hyperparameter optimization},
  booktitle = {Anais do XVI Workshop de Visão Computacional},
  location  = {Evento Online},
  year      = {2020},
  pages     = {104--110},
  publisher = {SBC},
  address   = {Porto Alegre, RS, Brasil},
  url       = {https://sol.sbc.org.br/index.php/wvc/article/view/13489}
}
```

## Authors

- **Erik Lucas da Rocha** — <erik.rocha@ufv.br>
- **Larissa Ferreira Rodrigues** — <larissa.f.rodrigues@ufv.br>
- **João Fernando Mari** — <joaof.mari@ufv.br>

Instituto de Ciências Exatas e Tecnológicas — Universidade Federal de Viçosa (UFV), Rio Paranaíba, MG, Brazil.

## Acknowledgments

- NVIDIA Corporation, for the donation of the TITAN Xp GPU used in this research.
- CAPES (Finance Code 001) and FAPEMIG, for financial support.

## License

Specify your license here (e.g., MIT, Apache 2.0). The PlantVillage dataset has its own license; verify it before redistributing data or derivative weights.
