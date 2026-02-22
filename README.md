# Deep Learning Classification of Skin Lesions Using the ISIC 2019 Challenge Dataset

DS 6050 | University of Virginia, School of Data Science

## Table of Contents
* [Team](#team)
* [Overview](#overview)
* [Dataset](#dataset)
* [Approach](#approach)
* [Technologies](#technologies)
* [Setup](#setup)
* [Project Status](#project-status)

## Team
* Efrain O. (dpy8wq)
* Haejin S. (umw7eg)
* Virginia B. (vrb9e)
* Bill L. (wpl3a)

## Overview
More than 5 million cases of skin cancer are diagnosed annually in the United States, with melanoma cases sharply increasing over the past few decades. Early detection of melanoma is associated with a 5-year survival rate of over 95%, creating a critical need for better diagnostic tools.

Inspired by the [ISIC](https://www.isic-archive.com/) benchmark challenges, this project explores how deep learning models can efficiently classify skin lesions with high sensitivity and specificity.

## Dataset
**ISIC 2019 Aggregate Dataset** — 33,569 JPG images and associated metadata compiled from three sources, pre-split into training and test sets. We use the 2019 data (rather than the current ISIC MILK10k) because published test ground truth is available for comprehensive model evaluation.

## Approach
We evaluate multiple CNN architectures — including **EfficientNet**, **ResNet**, and **MobileNet** — and down-select the highest performers for fine-tuning, ablation studies, and hyperparameter optimization using Optuna, W&B, and Hydra.

## Technologies
* Python 3.x
* PyTorch / Torchvision
* Albumentations
* Optuna
* Weights & Biases (W&B)
* Hydra

## Setup
```
$ git clone https://github.com/virginia-vrb9e/isic2019-skin-lesion-classification.git
$ cd isic2019-skin-lesion-classification
$ pip install -r requirements.txt
```

## Project Status
* 🟡 **Milestone 1** — Project proposal and literature review complete. Model development in progress.
* 🟡 **Milestone 2** — [upcoming: 3/22/2026]
* 🟡 **Final Deadline** — [upcoming: 5/17/2026]
