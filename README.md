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

## ⚙️ Installation

### Prerequisites
```bash
Python 3.8+  |  4GB RAM  |  NVIDIA GPU (optional)
```

### Quick Setup
```bash
# 1. Clone repository
git clone https://github.com/AI-WAJID/image_classification.git
cd image_classification

# 2. Install dependencies
pip install streamlit tensorflow opencv-python pillow numpy matplotlib pandas

# 3. Launch web app
streamlit run app.py
```

### Detailed Installation
```bash
# Create virtual environment
conda create -n cifar10 python=3.8
conda activate cifar10

# Install with specific versions
pip install streamlit==1.28.0 tensorflow==2.10.0 opencv-python==4.5.5.64 pillow==10.0.0 numpy==1.24.3 matplotlib==3.7.2 pandas==2.0.3

# For GPU support (optional)
# Ensure CUDA 11.2 and cuDNN 8.1 are installed
```

## 🚀 Usage

### Web Application (Recommended)
```bash
streamlit run app.py
# Open http://localhost:8501 in your browser
```

1. **Upload Image**: Click "Choose an image file"
2. **View Results**: See original + processed (32×32) images
3. **Analyze Predictions**: Check confidence scores and probability distribution
4. **Interactive Charts**: Explore prediction probabilities for all classes

### Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```
- **Training Pipeline**: Complete model development workflow
- **Data Analysis**: CIFAR-10 dataset exploration
- **Model Evaluation**: Performance metrics and visualizations
- **Experimentation**: Modify architecture and hyperparameters

### Python API
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('image_classification.h5')

# Predict single image
def predict_image(image_path):
    image = Image.open(image_path).resize((32, 32))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(image_array)
    return np.argmax(prediction[0]), np.max(prediction[0])

# Usage
class_id, confidence = predict_image('your_image.jpg')
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"Predicted: {classes[class_id]} ({confidence:.1%})")
```

## 📈 Performance

### Model Metrics
| Metric | Training | Validation | Test |
|---------|----------|------------|------|
| **Accuracy** | 95.2% | 89.3% | **87.8%** |
| **Loss** | 0.142 | 0.298 | 0.325 |
| **Parameters** | 2.4M trainable | | |

### Training Configuration
- **Epochs**: 50 with early stopping
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Augmentation**: Width/height shifts, horizontal flips
- **Training Time**: ~45 minutes (GTX 1650)

### GPU Performance
- **Inference Time**: ~5ms per image (GPU) vs 50ms (CPU)
- **Memory Usage**: 2.1GB VRAM (GTX 1650)
- **Batch Processing**: 32 images in ~20ms

## 🛠️ Technologies

### Core Stack
- **[TensorFlow 2.10](https://tensorflow.org/)**: Deep learning framework
- **[Streamlit](https://streamlit.io/)**: Interactive web applications
- **[OpenCV](https://opencv.org/)**: Computer vision preprocessing
- **[NumPy](https://numpy.org/)**: Numerical computing

### Development Tools
- **Jupyter Notebook**: Model experimentation
- **Python 3.8+**: Programming language
- **Anaconda**: Environment management
- **CUDA/cuDNN**: GPU acceleration

## 🔧 GPU Support

### NVIDIA GPU Setup
```python
# Automatic GPU configuration in app.py
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Requirements
- **NVIDIA GPU**: GTX 1050+ or RTX series
- **CUDA**: 11.2
- **cuDNN**: 8.1
- **VRAM**: 2GB+ recommended

## 🚀 Future Enhancements

### Planned Features
- [ ] **Transfer Learning**: ResNet50, EfficientNet integration
- [ ] **Batch Processing**: Multiple image upload support
- [ ] **Model Interpretability**: Grad-CAM visualizations
- [ ] **REST API**: Programmatic access endpoints
- [ ] **Mobile App**: TensorFlow Lite deployment

### Advanced Capabilities
- [ ] **Real-time Webcam**: Live video classification
- [ ] **Custom Datasets**: User dataset training pipeline
- [ ] **Model Comparison**: A/B testing different architectures
- [ ] **Cloud Deployment**: AWS/GCP integration

## 🤝 Contributing

Contributions welcome! Please check:

1. **Issues**: Report bugs or suggest features
2. **Pull Requests**: Submit improvements
3. **Documentation**: Help expand guides
4. **Testing**: Add validation scripts

```bash
# Development setup
git clone https://github.com/AI-WAJID/image_classification.git
cd image_classification
pip install -r requirements.txt
# Make changes and submit PR
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **CIFAR-10 Dataset**: Alex Krizhevsky, University of Toronto
- **TensorFlow Team**: Excellent deep learning framework
- **Streamlit Team**: Amazing web app development tool
- **Open Source Community**: For continuous innovation

## 📧 Contact

**Wajid Ali** - AI/ML Engineer

- **GitHub**: [@AI-WAJID](https://github.com/AI-WAJID)
- **Repository**: [image_classification](https://github.com/AI-WAJID/image_classification)
- **Issues**: [Report Bug](https://github.com/AI-WAJID/image_classification/issues)

---

<div align="center">

**⭐ Star this repo if it helped you!**

Made with ❤️ using TensorFlow & Streamlit

[⬆ Back to Top](#cifar-10-image-classification-with-tensorflow-and-streamlit)

</div>
