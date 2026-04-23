# Pneumonia Detection from Chest X-Ray Images

*This repository contains a machine learning course project focused on detecting pneumonia from chest X-ray images using both traditional machine learning models and deep learning approaches.*

## Overview

This project develops an automated system to classify chest X-ray images as Pneumonia or Normal.

The goal is to improve diagnostic efficiency by leveraging machine learning, reducing reliance on manual interpretation. The project explores both:
- Traditional ML models (Logistic Regression, SVM, Feedforward Neural Network)
- Deep learning models (Convolutional Neural Networks)

A key component of the project is comparing performance across these approaches and understanding trade-offs between feature-based methods and end-to-end learning.

## Dataset

The dataset used is the Chest X-Ray Pneumonia dataset from Kaggle:

Download here:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Structure

Structure

After extracting, the dataset should contain:
```
train/
val/
test/
```

Each folder contains:
```
PNEUMONIA/
NORMAL/
```

## Setup

*1. Create environment*
```
conda create -n pneumonia-detection python=3.10
conda activate pneumonia-detection
```
*2. Install dependencies*
```
conda install numpy matplotlib
pip install torch torchvision
```

## Running the Project

This project is implemented as a Jupyter Notebook.

*1. Launch notebook*
```
jupyter notebook
```
Open the notebook file in your browser.

*2. Prepare dataset*
- Download and extract the dataset from Kaggle
- Place the folders (train, val, test) in the same directory as the notebook

Your structure should look like:
```
project/
├── notebook.ipynb
├── train/
├── val/
├── test/
```

*3. Run all cells*

Run the notebook from top to bottom.

The notebook will:
- Set random seeds for reproducibility
- Detect GPU/CPU device automatically
- Preprocess images (resize to 224x224, normalize using ImageNet stats)
- Apply data augmentation on training data
- Merge datasets and create new splits
- Extract features using pretrained models
- Train and evaluate:
    - Logistic Regression
    - SVM
    - Feedforward Neural Network
    - CNN

## Key Implementation Details

*Preprocessing*

- Resize all images to 224 × 224
- Normalize using ImageNet mean and standard deviation
- Apply augmentation:
    - Horizontal flip
    - Rotation
    - Brightness/contrast adjustments

*Feature Extraction*
- Used ResNet18 (pretrained) to extract 512-dimensional feature vectors
- Converted image data into tabular format for traditional ML models

*Models Implemented*
- Majority-class baseline
- Logistic Regression
- Support Vector Machine (RBF kernel)
- Feedforward Neural Network (PyTorch)
- Convolutional Neural Network (CNN)

## Evaluation

Models were evaluated using:

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

The CNN achieved the strongest performance, outperforming all baseline models by learning spatial features directly from images.

## Notes
- The original dataset split was adjusted to 60-20-20 (train/val/test) for better validation
- GPU is recommended for CNN training but not required
- Ensure all dataset folders are correctly placed before running

## References
- Kaggle Dataset: Chest X-Ray Pneumonia
- PyTorch Documentation
- scikit-learn Documentation
- Research papers on pneumonia detection (CheXNet, etc.)