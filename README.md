# Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles

[![arXiv](https://img.shields.io/badge/arXiv-2601.22068-b31b1b.svg)](https://arxiv.org/abs/2601.22068)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/Conferences/2026)

Official repository for **Singular Value Ensemble (SVE)**, a parameter-efficient approach for uncertainty quantification in pretrained foundation models.

**Mehmet Ozgur Turkoglu¹, Dominik J. Mühlematter², Alexander Becker², Konrad Schindler², Helge Aasen¹**
¹ Agroscope, Earth Observation of Agroecosystems
² ETH Zürich, Photogrammetry and Remote Sensing

📄 **Paper:** [arXiv:2601.22068](https://arxiv.org/abs/2601.22068)
🏛️ **Accepted at ICML 2026**

> **Code release:** implementation and reproducibility scripts will be added to this repository.

---

## TL;DR

Foundation models are powerful, but their predictions can still be **overconfident and poorly calibrated**. Deep ensembles provide strong uncertainty estimates, but require training and storing multiple complete models.

**Singular Value Ensemble (SVE)** turns a single pretrained foundation model into an ensemble by sharing its pretrained singular-vector basis and learning only lightweight, **member-specific singular values**.

SVE provides:

* **<1% parameter overhead** in typical settings,
* strong predictive accuracy and calibration,
* meaningful epistemic uncertainty under distribution shift,
* competitive OOD detection,
* applicability across **vision and language models ranging from 22M to 7B parameters**.

---

## Method

<p align="center">
  <img src="assets/sve_method.png" width="95%" alt="Singular Value Ensemble method">
</p>

<p align="center">
  <em>Singular Value Ensemble shares the pretrained singular-vector basis across ensemble members while allowing each member to learn its own singular values.</em>
</p>

For a pretrained weight matrix \(W\), we first consider its singular value decomposition:

$$
W = U \Sigma V^\top.
$$

Instead of learning a complete copy of \(W\) for every ensemble member, SVE keeps the singular vectors \(U\) and \(V\) shared and creates member-specific singular values:

$$
W^{(m)} = U \Sigma^{(m)} V^\top,
$$

where \(m \in \{1,\ldots,M\}\) denotes the ensemble member.

The key idea is simple:

* **\(U\) and \(V\)** preserve the shared representation learned during pretraining.
* **\(\Sigma^{(m)}\)** determines how strongly each ensemble member uses the corresponding pretrained directions.
* Each member therefore learns a different reweighting of the same pretrained feature basis.

To break symmetry between ensemble members while remaining close to the pretrained solution, the singular values are initialized with small perturbations:

$$
\Sigma^{(m)}
=
\Sigma \odot
\left(1+\epsilon^{(m)}\right),
\qquad
\epsilon^{(m)}
\sim
\mathcal{N}(0,\sigma_{\mathrm{init}}^2 I).
$$

The members are then optimized jointly. At inference time, predictions are combined exactly as in a conventional ensemble.

---

## Why Singular Values?

A pretrained model already contains a rich set of learned directions.

Rather than asking every ensemble member to relearn an entirely new representation, SVE asks a much smaller question:

> **How strongly should each member use the directions already learned during pretraining?**

The singular vectors define the shared basis, while the singular values control the importance of individual directions.

This creates diversity between ensemble members without duplicating the complete model.

---

## Deep Ensemble vs. SVE

A conventional deep ensemble independently learns \(M\) complete models:

$$
W^{(1)}, W^{(2)}, \ldots, W^{(M)}.
$$

SVE instead shares the expensive part:

$$
U,\;V
$$

and only learns

$$
\Sigma^{(1)},\Sigma^{(2)},\ldots,\Sigma^{(M)}.
$$

For a matrix

$$
W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}},
$$

a full ensemble member requires

$$
\mathcal{O}
\left(
d_{\text{out}}d_{\text{in}}
\right)
$$

parameters, whereas SVE requires only

$$
\mathcal{O}
\left(
\min(d_{\text{out}},d_{\text{in}})
\right)
$$

member-specific parameters.

| Method                      | Member-specific parameters |
| --------------------------- | -------------------------: |
| Deep Ensemble               |                 Full model |
| LoRA Ensemble               |          Low-rank matrices |
| **Singular Value Ensemble** |   **Singular values only** |

This difference becomes increasingly important as foundation models grow.

---

## Key Results

SVE is evaluated across vision and language models, covering calibration, uncertainty estimation, out-of-distribution detection, and robustness under dataset shift.

### Vision

On **Flowers102 with DINO ViT-S/16**, SVE achieves strong accuracy while maintaining competitive calibration:

| Method        | Accuracy ↑ |   ECE ↓ |
| ------------- | ---------: | ------: |
| Deep Ensemble |       91.5 | **0.9** |
| LoRA Ensemble |       94.6 |     1.1 |
| **SVE**       |   **95.4** |     1.0 |

---

### Language Models

SVE also scales to large language models.

On **ARC-Easy with LLaMA-2-7B**:

| Method        | Accuracy ↑ |   ECE ↓ |
| ------------- | ---------: | ------: |
| Deep Ensemble |       85.8 |     9.9 |
| LoRA Ensemble |       86.0 |     9.0 |
| **SVE**       |       85.8 | **3.8** |

SVE substantially improves calibration while avoiding multiple full copies of a 7B-parameter model.

---

## Out-of-Distribution Detection

An important requirement for uncertainty estimation is that model disagreement increases when inputs move outside the training distribution.

Using **CIFAR-100 as in-distribution data and CIFAR-10 as OOD data**, SVE produces meaningful ensemble disagreement and competitive OOD detection.

| Method        | AUROC using Mutual Information ↑ |
| ------------- | -------------------------------: |
| Deep Ensemble |                             79.2 |
| LoRA Ensemble |                         **82.8** |
| **SVE**       |                             81.6 |

The results show that diversity created purely through singular-value adaptation can capture useful epistemic uncertainty.

---

## Robustness Under Distribution Shift

SVE is also evaluated on **CIFAR-100-C**, where images are progressively corrupted.

As corruption severity increases:

* predictive uncertainty increases,
* ensemble disagreement increases,
* calibration remains competitive,
* SVE retains substantially lower parameter cost than conventional deep ensembles.

This indicates that SVE uncertainty responds meaningfully to increasingly difficult and shifted inputs.

---

## Parameter Efficiency

The main advantage of SVE becomes especially visible for large pretrained models.

For **BERT-base with an 8-member ensemble**:

| Method        | Additional Parameters |
| ------------- | --------------------: |
| Deep Ensemble |                 ~700% |
| LoRA Ensemble |                ~10.3% |
| **SVE**       |             **~1.0%** |

SVE therefore provides ensemble-style uncertainty estimation without requiring multiple complete copies of the foundation model.

---

## The Role of Pretraining

SVE is specifically designed for **pretrained models**.

Its effectiveness relies on the assumption that the pretrained singular-vector basis already contains useful directions for the downstream task.

The experiments show that SVE becomes increasingly competitive as the quality of pretraining improves.

This suggests an important interpretation:

> **Strong foundation models may not require multiple independently learned representations to obtain useful ensemble uncertainty. Diverse reweightings of a shared pretrained basis can already provide strong uncertainty estimates.**

---

## What Does SVE Estimate?

Given predictions from \(M\) ensemble members,

$$
p_m(y\mid x),
$$

the ensemble prediction is

$$
p(y\mid x)
=
\frac{1}{M}
\sum_{m=1}^{M}
p_m(y\mid x).
$$

Member disagreement can then be used to quantify **epistemic uncertainty**.

In the paper, we evaluate uncertainty using metrics including:

* Expected Calibration Error (**ECE**),
* Negative Log-Likelihood (**NLL**),
* Brier score,
* predictive entropy,
* mutual information,
* OOD AUROC,
* robustness under corruption severity.

---

## Evaluated Models

### Vision

* DINO ViT-S/16
* DINOv2 ViT-S/14

### Language

* BERT-base
* LLaMA-2-7B

The experiments therefore cover models ranging from approximately **22 million to 7 billion parameters**.

---

## Benchmarks

### Vision

* Flowers102
* CIFAR-100
* Describable Textures Dataset (DTD)
* Oxford-IIIT Pets
* CIFAR-100-C
* CIFAR-10 for OOD evaluation

### NLP

* SST-2
* ARC-Easy

---

## When Is SVE Useful?

SVE is particularly useful when:

* a strong pretrained backbone is already available,
* reliable uncertainty estimates are required,
* deep ensembles are prohibitively expensive,
* parameter-efficient adaptation is important,
* calibration matters,
* robustness under distribution shift matters,
* foundation models are too large to duplicate several times.

---

## Repository Status

Implementation and reproducibility scripts will be released in this repository.

The repository will include:

* SVE layers and model wrappers,
* vision experiments,
* NLP experiments,
* calibration metrics,
* epistemic uncertainty metrics,
* OOD evaluation,
* corruption robustness experiments,
* experiment configurations.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{turkoglu2026singular,
  title     = {Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles},
  author    = {Turkoglu, Mehmet Ozgur and M{\"u}hlematter, Dominik J. and Becker, Alexander and Schindler, Konrad and Aasen, Helge},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

For the arXiv version:

```bibtex
@article{turkoglu2026singular,
  title   = {Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles},
  author  = {Turkoglu, Mehmet Ozgur and M{\"u}hlematter, Dominik J. and Becker, Alexander and Schindler, Konrad and Aasen, Helge},
  journal = {arXiv preprint arXiv:2601.22068},
  year    = {2026}
}
```

---

## Links

* 📄 [Paper](https://arxiv.org/abs/2601.22068)
* 📑 [PDF](https://arxiv.org/pdf/2601.22068)
* 🏛️ [ICML 2026](https://icml.cc/Conferences/2026)
