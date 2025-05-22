# 🧠 Skin Lesion Classification with CNN on Bi-Dimensional Signals

A convolutional neural network (CNN) built from scratch to classify medical images of skin lesions using a subset of the Skin Cancer MNIST: HAM10000 dataset. The goal is to distinguish between benign and malignant lesions through pattern recognition in bi-dimensional image data.

## 📌 Overview

This project is part of a machine learning challenge focused on bi-dimensional signal classification. In the second task, we addressed the problem of classifying medical images from the HAM10000 dataset using a custom-designed CNN implemented from the ground up, without relying on deep learning frameworks.

## 🧬 Dataset

- **Source**: [Skin Cancer MNIST: HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Note**: Due to computational and memory limitations, the classification task was reduced to **4 out of the original 7 classes**.
- A **balanced subset** of the data was selected to ensure fair class representation and efficient training.

## 🧠 Model Architecture

The CNN consists of:

- Multiple **convolutional layers** to extract relevant features from skin lesion images.
- **Pooling layers** to reduce dimensionality and computational cost.
- **Fully connected layers** for final classification.

The network was trained from scratch using low-level libraries (e.g., NumPy), without using high-level frameworks such as TensorFlow or PyTorch.

## 🏋️‍♂️ Training Strategy

- The dataset was split into training and validation sets.
- **Data augmentation** techniques were applied to increase dataset diversity and improve model generalization.
- Training performance was monitored through:
  - Accuracy and loss plots
  - Validation metrics to avoid overfitting

## 📊 Results

- Training and validation accuracy improved steadily during training.
- Visual diagnostics (loss/accuracy plots) were used to guide hyperparameter tuning and training time.

## 🛠 Tools Used

- Python (NumPy, Matplotlib)
- Custom CNN implementation (no deep learning libraries)
- Jupyter Notebook / VS Code (recommended IDEs)

