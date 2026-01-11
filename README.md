# Deep Learning Image Classification

A collection of deep learning projects for image classification tasks, including celebrity face recognition and flower species classification using transfer learning and custom CNN architectures.

## Projects Overview

### 1. Celebrity Face Recognition (14 celeb/)

Face recognition system trained on 14 celebrity faces using transfer learning.

**Notebooks:**
- `14_faces_Densenet201.ipynb` - Main training notebook using DenseNet201 pretrained on ImageNet
- `augment.ipynb` - Data augmentation pipeline using the Augmentor library (rotation, zoom, flip, distortion)
- `load.ipynb` - Data loading and preprocessing utilities

**Key Features:**
- Transfer learning with DenseNet201 (18M+ parameters)
- Image preprocessing and resizing to 224x224
- Data augmentation for improved generalization
- Training/validation split for model evaluation

---

### 2. Flower Classification (flower detection/)

Large-scale flower classification on 104 different flower species using TPU acceleration.

**Notebook:**
- `Elementary.ipynb` - Comprehensive notebook comparing multiple architectures

**Models Compared:**
| Model | Parameters | Architecture |
|-------|------------|--------------|
| DenseNet201 | 18.5M | Pre-trained, fine-tuned |
| Xception | 21M | Pre-trained, fine-tuned |
| ResNet50 | 23.8M | Pre-trained, fine-tuned |
| Custom CNN (3 layers) | 8.4M | Built from scratch |
| Custom VGG (3 blocks) | 8.5M | Built from scratch |

**Key Features:**
- TPU distributed training (8 replicas)
- TFRecord data pipeline for efficient loading
- Multiple data augmentation strategies (v1, v2, v3)
- Learning rate scheduling with exponential decay
- Class weight balancing for imbalanced data
- Confusion matrix and F1 score evaluation
- Kaggle competition submission format

---

### 3. MS-Celeb-1M Face Recognition (Ms celeb/)

Large-scale face recognition using the MS-Celeb-1M dataset with distributed data processing.

**Notebook:**
- `ms_celeb_1GB.ipynb` - Training pipeline for large-scale face recognition

**Key Features:**
- MobileNetV3Small backbone for efficient inference
- PySpark for distributed data loading
- Dask for parallel CSV processing
- Base64 image decoding from CSV format
- Custom data generator for memory-efficient training
- 1,201 identity classes

---

## Technologies Used

- **Deep Learning:** TensorFlow 2.x, Keras
- **Pre-trained Models:** DenseNet201, Xception, ResNet50, MobileNetV3
- **Data Processing:** OpenCV, PIL, NumPy, Pandas
- **Distributed Computing:** PySpark, Dask
- **Data Augmentation:** Augmentor, TensorFlow image ops
- **Hardware Acceleration:** TPU v3-8, GPU (Google Colab)
- **Evaluation:** Scikit-learn (confusion matrix, F1, precision, recall)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

2. Install dependencies:
```bash
pip install tensorflow keras opencv-python pillow numpy pandas scikit-learn matplotlib seaborn augmentor
```

3. For large-scale processing (MS-Celeb):
```bash
pip install pyspark dask
```

## Usage

Each notebook is designed to run in Google Colab with GPU/TPU acceleration. Upload to Colab and run cells sequentially.

**Note:** Some notebooks require datasets to be stored in Google Drive or Kaggle datasets.

## Authors

- Ali Babaloo
- Arshia Tehrani
- Pouya Sharifi
- Pouya Ebrahimi
