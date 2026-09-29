# Singular Value Ensembles for Foundation Model Uncertainty

[![arXiv](https://img.shields.io/badge/arXiv-2601.22068-b31b1b.svg)](https://arxiv.org/abs/2601.22068)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/Conferences/2026)

Official repository for **"Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles"**, accepted at **ICML 2026**.

**Mehmet Ozgur Turkoglu, Dominik J. Mühlematter, Alexander Becker, Konrad Schindler, Helge Aasen**

📄 [Paper](https://arxiv.org/abs/2601.22068)

---

## TL;DR

Deep ensembles are among the most reliable approaches for uncertainty estimation, but they require training and storing multiple complete models.

**Singular Value Ensemble (SVE)** turns a single pretrained foundation model into a parameter-efficient ensemble by sharing the pretrained weight structure and learning only **member-specific singular values**.

This provides ensemble-style uncertainty estimation with **less than 1% additional parameters** in typical settings.

---

## Method

<p align="center">
  <img src="assets/sve_method.jpg" width="95%" alt="Singular Value Ensemble method">
</p>

For a pretrained weight matrix:

$$
W = U \Sigma V^\top
$$

SVE keeps the singular-vector basis **U** and **V** shared across the ensemble, while each member learns its own singular values:

$$
W^{(m)} = U \Sigma^{(m)} V^\top
$$

where \(m\) denotes the ensemble member.

In other words:

* **U and V are shared** and preserve the representation learned during pretraining.
* **Singular values are member-specific** and determine how strongly each member uses the pretrained directions.
* Different singular-value configurations create diverse predictors without duplicating the complete foundation model.

The resulting predictions are combined in the same way as a standard ensemble.

---

## Why SVE?

A conventional deep ensemble stores a complete model for every member.

SVE instead shares almost the entire pretrained model and only introduces a small number of member-specific parameters.

This makes uncertainty estimation practical for large pretrained models, including foundation models with billions of parameters.

---

## Getting Started

This repository contains a minimal implementation of SVE for image classification with a pretrained DINO ViT.

### Installation

```bash
git clone https://github.com/moturkoglu/Singular-Value-Ensemble.git
cd Singular-Value-Ensemble
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

This trains a 4-member SVE on Oxford Flowers-102 with `vit_small_patch16_224.dino`. The dataset is downloaded to `./data` automatically.

Other datasets:

```bash
python train.py --dataset {flowers102,dtd,aircraft,pets,food101,cifar100}
```

The checkpoint with the best validation accuracy is evaluated on the test set. Accuracy, NLL, Brier score and ECE are reported before and after temperature scaling (temperature fit on the validation set).

### Default settings

| Argument | Default |
|---|---|
| `--backbone` | `vit_small_patch16_224.dino` |
| `--n_members` | 4 |
| `--epochs` / `--warmup_epochs` | 10 / 5 |
| `--lr` | 1e-3 (AdamW, cosine schedule) |
| `--weight_decay` | 0.05 |
| `--batch_size` | 16 |
| `--init_std` | 0.01 |

### Files

- `sve.py`: SVE layer, member-specific heads and model
- `train.py`: data, training and evaluation

---

## Main Results

SVE is evaluated across vision and language models ranging from **22M to 7B parameters**.

Some representative results:

| Setting                        | Result                       |
| ------------------------------ | ---------------------------- |
| **DINO ViT-S/16 · Flowers102** | **95.4% accuracy, 1.0% ECE** |
| **LLaMA-2-7B · ARC-Easy**      | **85.8% accuracy, 3.8% ECE** |
| **CIFAR-100 → CIFAR-10 OOD**   | **81.6 AUROC**               |
| **BERT-base · 8 members**      | **~1% parameter overhead**   |

Across the experiments, SVE provides competitive calibration, epistemic uncertainty, OOD detection, and robustness while remaining substantially more parameter-efficient than conventional deep ensembles.

---

## Key Idea

> **Foundation models already provide a strong feature basis. Instead of training several complete models, SVE creates an ensemble by learning different reweightings of this shared pretrained basis.**

---

## Citation

```bibtex
@inproceedings{turkoglu2026singular,
  title     = {Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles},
  author    = {Turkoglu, Mehmet Ozgur and M{\"u}hlematter, Dominik J. and Becker, Alexander and Schindler, Konrad and Aasen, Helge},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
