# 🤟 ASL Recognition System

Machine Learning based American Sign Language gesture recognition using CNNs and ensemble learning

An end to end computer vision project that detects and classifies American Sign Language hand gestures from images using deep learning models and an ensemble voting strategy to improve prediction confidence and reduce misclassification.

## 🎯 Overview

This project addresses the communication gap between American Sign Language users and individuals unfamiliar with sign language by building a machine learning based gesture recognition system. The system uses image data of ASL alphabets and numbers and applies convolutional neural networks for feature extraction and classification. Multiple deep learning models are evaluated, and an ensemble voting classifier is used to combine their predictions for higher reliability and robustness.

## ✨ Key Highlights

* Trained and evaluated multiple CNN based architectures for ASL gesture recognition
* VGG16 emerged as the strongest individual model across datasets
* Ensemble voting classifier improves prediction confidence and reduces misclassification
* Robust preprocessing pipeline including background removal and data augmentation
* Multi class classification covering ASL alphabets and numeric gestures
* Deployed model for real time prediction through a simple web interface

## 📋 Features

### Analysis Components

| Component           | Description                            | Status   |
| ------------------- | -------------------------------------- | -------- |
| Dataset Preparation | Curated and balanced ASL image dataset | Complete |
| Preprocessing       | Background removal and augmentation    | Complete |
| VGG16 Model         | Transfer learning based CNN            | Complete |
| ResNet50 Model      | Deep residual network for comparison   | Complete |
| Custom CNN          | Lightweight CNN architecture           | Complete |
| Ensemble Classifier | Average voting across models           | Complete |
| Model Evaluation    | Accuracy, loss, confusion matrices     | Complete |
| Deployment          | Flask based inference interface        | Complete |

## 🧠 Methodology

### 1. Data Collection

* ASL image dataset containing alphabets and digits
* Multiple samples per class to ensure diversity

### 2. Exploratory Analysis

* Class distribution analysis
* Visual inspection of gesture variability

### 3. Data Preprocessing

* Background removal to isolate hand gestures
* Image rescaling and normalization
* Data augmentation including rotation, flipping, and shearing

### 4. Core Modeling

* VGG16 with frozen base layers for feature extraction
* ResNet50 for deep residual learning comparison
* Custom CNN for baseline performance
* Ensemble voting classifier combining all three models

### 5. Validation and Evaluation

* Training and validation performance monitoring
* Confusion matrix and classification reports
* Overfitting control using early stopping and learning rate scheduling

## 📊 Key Results

### Performance Outcomes

* VGG16 consistently delivered the strongest standalone performance
* Ensemble voting reduced class level confusion
* Models generalized well to unseen test data
* Stable convergence observed across training cycles

### Business and Real World Insights

* Ensemble learning increases trust in predictions for assistive systems
* Background removal significantly improves gesture clarity
* System demonstrates feasibility for real time accessibility tools
* Provides a foundation for future video based and live camera recognition

## 🛠️ Technologies

### Tools

| Technology | Purpose                         |
| ---------- | ------------------------------- |
| Python     | Core development language       |
| TensorFlow | Deep learning framework         |
| Keras      | Model construction and training |
| OpenCV     | Image processing                |
| NumPy      | Numerical computation           |
| Flask      | Model deployment                |

### Libraries Used

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
import numpy as np
```

## 💡 Skills Demonstrated

* Computer vision and image preprocessing
* Deep learning with CNN architectures
* Transfer learning and ensemble modeling
* Model evaluation and overfitting control
* End to end ML pipeline design
* Model deployment using Flask

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* TensorFlow
* OpenCV
* NumPy

### Installation

```bash
git clone https://github.com/your-username/ASL-Recognition.git
cd ASL-Recognition
pip install -r requirements.txt
```

### Usage

```bash
python train.py
python app.py
```

Upload an image through the web interface to get ASL gesture predictions.
