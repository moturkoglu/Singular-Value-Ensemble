# Singular Value Ensembles for Foundation Model Uncertainty

[![arXiv](https://img.shields.io/badge/arXiv-2601.22068-b31b1b.svg)](https://arxiv.org/abs/2601.22068)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/Conferences/2026)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official implementation of **"Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles"** (ICML 2026).

Mehmet Ozgur Turkoglu¹, Dominik J. Mühlematter², Alexander Becker², Konrad Schindler², Helge Aasen¹  
¹ Agroscope, Earth Observation of Agroecosystems &nbsp;&nbsp; ² ETH Zürich, Photogrammetry & Remote Sensing

📄 [Paper (arXiv)](https://arxiv.org/abs/2601.22068) &nbsp;|&nbsp; 🏛️ [ICML 2026](https://icml.cc/Conferences/2026)

---

**TL;DR.** Singular Value Ensemble (SVE) turns a single pretrained foundation model into an implicit ensemble by freezing the SVD basis (**U**, **V**) of its weight matrices and learning only per-member singular values. This adds less than 1% to the parameter count while matching deep-ensemble calibration, OOD detection, and shift robustness across vision and NLP backbones from 22M to 7B parameters.
