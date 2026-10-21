# chest-xray-pneumonia-resnet18
Chest X-ray pneumonia classification using PyTorch, ResNet18 transfer learning, and data augmentation.
# Chest X-Ray Pneumonia Classification with ResNet18

A deep learning project using PyTorch and transfer learning to classify chest X-ray images as **NORMAL** or **PNEUMONIA**.

## Project Overview

This project explores how convolutional neural networks can be applied to medical image classification. I used a pretrained ResNet18 model and compared a baseline experiment with an augmented version to investigate the effect of data augmentation on classification performance.

The project focuses on model development, data preprocessing, class imbalance, validation, and evaluation on a held-out test set.

## Objectives

- Build a binary image classifier using PyTorch.
- Apply transfer learning with a pretrained ResNet18 architecture.
- Compare baseline training with training that uses image augmentation.
- Address class imbalance using class-weighted cross-entropy loss.
- Evaluate performance using accuracy, ROC-AUC, precision, recall, F1-score, and confusion matrices.
- Investigate model limitations and generalization to unseen images.

## Methodology

### 1. Data Preparation

- Load and inspect the chest X-ray dataset.
- Prepare image paths and binary class labels.
- Resize images and apply the transformations required by the model.
- Split data into training, validation, and test sets, using group-based splitting where implemented to reduce overlap between related images.

### 2. Model Architecture

**ResNet18 with transfer learning**

The project uses an ImageNet-pretrained ResNet18 backbone adapted for binary classification.

### 3. Experiments

**Baseline model**
- Train the pretrained ResNet18 using the baseline image transformations.
- Evaluate performance on validation and test data.

**Augmented model**
- Apply mild image augmentation, including rotation and brightness/contrast variation.
- Use class-weighted cross-entropy loss to address class imbalance.
- Select the best checkpoint using validation ROC-AUC.
- Evaluate the selected checkpoint on the held-out test set.

## Results

The following are the recorded test results from the current notebook.

| Metric | Baseline | Augmented model |
|---|---:|
