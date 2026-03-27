# Breast Cancer Detection in Ultrasound Images
### Hybrid CNN-ViT Architecture with Explainable AI

Diploma Thesis — Electrical & Computer Engineering, Democritus University of Thrace  
🏆 **2nd Prize — Student Competition, European Symposium on Biomedical Engineering (2025)**

---

## Overview

This project presents a deep learning system for automatic breast cancer 
diagnosis from ultrasound images. The core model is **HoVerTrans** - a hybrid 
architecture that combines Convolutional Neural Networks (CNNs) and Vision 
Transformers (ViT) - extended with two novel additions:

- **DenseNet encoder** for richer multi-scale feature extraction
- **Superpixel Cross-Attention (SCA)** for improved local detail preservation

Explainability methods (Grad-CAM++ and LIME) are applied to make model 
decisions interpretable for clinical use.

---

## Results

| Method | Accuracy | Recall | Precision | F1 Score | AUC |
|---|---|---|---|---|---|
| Baseline HoVerTrans | 0.879 | 0.925 | 0.890 | 0.906 | 0.921 |
| + DenseNet encoder | 0.881 | 0.922 | 0.893 | 0.908 | 0.918 |
| + Token Selection | 0.877 | 0.899 | 0.907 | 0.902 | 0.916 |
| + SCA Attention | **0.886** | 0.924 | 0.898 | **0.911** | 0.908 |

> Best overall balance achieved with **DenseNet + SCA**, combining 
> fine-grained local detail with broad contextual understanding.

---

## Architecture

The model builds on **HoVerTrans**, which processes ultrasound images through:

1. **Convolutional Stem** — initial feature extraction
2. **Patch + Strip Embeddings** — horizontal and vertical spatial encoding 
   (reflecting how benign tumors grow within tissue layers vs. malignant ones 
   that penetrate across layers)
3. **HoVer-Transformer Blocks** — four-branch attention (H, V, H2V, V2H) 
   capturing both intra-layer and inter-layer spatial relationships
4. **Convolutional Merging** — fuses H2V and V2H branch features

**Added innovations:**
- DenseNet encoder replacing the original backbone
- SuperPixel Cross-Attention (SCA) module inserted before feature merging

---

## Explainability

### Grad-CAM++
Highlights the image regions most responsible for the model's prediction, 
using second and third-order gradients for higher spatial precision than 
standard Grad-CAM.
<img width="1243" height="287" alt="image" src="https://github.com/user-attachments/assets/1f49d1d5-bc18-405e-bd60-a24be9556685" />


### LIME
Segments the input image into superpixels and trains a local linear model 
to identify which regions positively or negatively influence the classification.

<img width="1334" height="814" alt="image" src="https://github.com/user-attachments/assets/4b75a8b2-e38e-4052-83bc-173b9289036b" />


---


---

## Stack

| | |
|---|---|
| Framework | PyTorch |
| Explainability | GradCAM++, LIME |
| Base model | HoVerTrans |
| Language | Python 3.x |

---


## References

1. Shi, Z., Lin, J., Zhao, B., Huang, C., Qiu, B., Cui, Y., & Liu, M. (2023).
   HoVer-Trans: Anatomy-Aware HoVer-Transformer for ROI-Free Breast Cancer
   Diagnosis in Ultrasound Images. *IEEE Transactions on Medical Imaging*,
   42(6), 1696–1706. https://doi.org/10.1109/TMI.2023.3236011

2. Mei, J., Chen, L.-C., & Yuille, A. (2024). SPFormer: Enhancing Vision
   Transformer with Superpixel Representation. arXiv:2401.02931.
   https://arxiv.org/abs/2401.02931
