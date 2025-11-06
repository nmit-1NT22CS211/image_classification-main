# CIFAR-10 Image Classification with TensorFlow and Streamlit

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

*A deep learning project that classifies images into 10 different categories using Convolutional Neural Networks*

[Demo](#-demo) • [Features](#-features) • [Installation](#%EF%B8%8F-installation) • [Usage](#-usage) • [Architecture](#%EF%B8%8F-model-architecture) • [Results](#-performance)

</div>

---

## 🔍 Overview

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset** into 10 distinct categories. Built with **TensorFlow/Keras** and deployed through an interactive **Streamlit web application** for real-time predictions with confidence scores.

### Key Highlights
- 🎯 **87.8% Test Accuracy** on CIFAR-10 dataset
- ⚡ **GPU-Accelerated** inference with CUDA support
- 🖼️ **Real-time Classification** via Streamlit web interface
- 📊 **Interactive Visualizations** with confidence scores
- 🚀 **Easy Deployment** with single command setup

## 📁 Project Structure

```
image_classification/
│
├── 🤖 Core Model
│   ├── notebook.ipynb                 # Complete training pipeline & experiments
│   └── image_classification.h5        # Trained CNN model (ready-to-use)
│
└── 🌐 Web Application
    └── app.py                         # Streamlit web interface
```

### File Descriptions

| File | Size | Description |
|------|------|-------------|
| `notebook.ipynb` | ~400KB | **Training Pipeline**: Data loading, CNN architecture, training with augmentation, evaluation |
| `image_classification.h5` | ~10MB | **Trained Model**: Pre-trained CNN ready for inference |
| `app.py` | ~8KB | **Web Interface**: Streamlit app with upload, predict, and visualization features |

## ✨ Features

### 🎯 Core Functionality
- **Real-time Classification**: Upload images and get instant predictions
- **Multi-format Support**: PNG, JPG, JPEG, BMP, TIFF compatibility
- **Confidence Visualization**: Interactive probability bars for all 10 classes
- **Image Preprocessing**: Automatic resize (32×32) and normalization
- **GPU Acceleration**: Optimized for NVIDIA GPU inference

### 📊 Advanced Features
- **Batch Normalization**: Faster convergence and better stability
- **Data Augmentation**: Width/height shifts, horizontal flips
- **Dropout Regularization**: Prevents overfitting (20% dropout)
- **Professional UI**: Clean Streamlit interface with custom CSS
- **Error Handling**: Robust model loading and prediction validation

## 📊 Dataset & Classes

**CIFAR-10 Dataset**: 60,000 32×32 color images across 10 categories

| Class | Examples | Class | Examples |
|-------|----------|-------|----------|
| ✈️ Airplane | Aircraft, jets | 🚢 Ship | Boats, vessels |
| 🚗 Automobile | Cars, buses | 🚛 Truck | Large vehicles |
| 🐦 Bird | Various species | 🐱 Cat | Domestic cats |
| 🦌 Deer | Wild deer | 🐶 Dog | Dog breeds |
| 🐸 Frog | Amphibians | 🐴 Horse | Horses, ponies |

## 🏗️ Model Architecture

### CNN Architecture
```python
Sequential CNN with 2.4M parameters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input (32×32×3) → Conv2D(32) + BatchNorm → Conv2D(32) + BatchNorm → MaxPool2D
                ↓
Conv2D(64) + BatchNorm → Conv2D(64) + BatchNorm → MaxPool2D
                ↓  
Conv2D(128) + BatchNorm → Conv2D(128) + BatchNorm → MaxPool2D
                ↓
Flatten → Dropout(0.2) → Dense(1024) → Dropout(0.2) → Dense(10)
```

### Key Design Choices
- **Progressive Filters**: 32 → 64 → 128 feature maps
- **Batch Normalization**: After each convolution for faster training
- **MaxPooling**: Reduces spatial dimensions and computational load
- **Dropout**: Prevents overfitting in dense layers
- **Adam Optimizer**: Adaptive learning rates for efficient training

