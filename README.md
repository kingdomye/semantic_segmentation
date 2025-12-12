# 🖼️ Semantic Segmentation Learning Project

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A PyTorch-based semantic segmentation project implementing multiple classic models for pixel-level image classification.

---

## 🛠️ Technology Stack

- **Python 3.10**
- **PyTorch**
- **torchvision**
- **matplotlib**
- **PIL (Pillow)**

---

## 🧠 Models

### 1. UNet
> A classic encoder-decoder architecture widely used in medical and natural image segmentation tasks.

**✨ Key Features**:
- **Encoder**: 5-layer convolutional network with ReLU activation and max pooling for downsampling
- **Decoder**: 4-layer transposed convolutional network for upsampling
- **Skip connections**: Concatenates encoder feature maps with corresponding decoder layers to preserve spatial information

**⚙️ Parameters**:
- `in_channels`: Input image channels (default: 1, set to 3 for RGB images)
- `num_classes`: Number of segmentation classes

### 2. FCN8s
> The first end-to-end fully convolutional network for semantic segmentation.

**✨ Key Features**:
- Based on **VGG16** pre-trained model as encoder
- Uses transposed convolution for upsampling
- Fuses multi-layer feature maps (pool3, pool4, fc7) to improve segmentation accuracy

**⚙️ Parameters**:
- `num_classes`: Number of segmentation classes

---

## 📂 Datasets

### VOC2012
The Visual Object Classes 2012 dataset contains 20 object classes plus 1 background class, totaling 21 classes.

- **Download URL**: [Baidu AI Studio - VOC2012](https://aistudio.baidu.com/datasetdetail/159243)

**Directory Structure**:
```text
VOC2012/ 
├── JPEGImages/          # Original RGB images (Input X) 
├── SegmentationClass/   # Ground truth segmentation masks (Output Y) 
└── ImageSets/ 
    └── Segmentation/ 
        ├── train.txt    # Training set file list 
        ├── val.txt      # Validation set file list 
        └── trainval.txt # Training-validation set file list
```

### SOD (Salient Object Detection)
The SOD dataset is used for salient object detection tasks, containing numerous annotated salient object images.

**Directory Structure**:
```text
SOD/ 
└── Gt/                  # Ground truth salient object masks
```

---

## 🚀 Quick Start

### Environment Setup

1. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate     # Windows
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Usage Examples

#### 🔹 UNet Model
```python
from src.UNet.UNet import UNet
from src.VocDataset import VOCSegmentDataset

# Load dataset
root = 'datasets/VOC2012'
dataset = VOCSegmentDataset(root_dir=root, image_set='train')

# Initialize model
model = UNet(num_classes=21, in_channels=3)  # 21-class segmentation, RGB input

# Test model
test_image, test_mask = dataset[0]
output = model(test_image.unsqueeze(0))  # Add batch dimension
```

#### 🔹 FCN8s Model
```python
from src.FCN.FCN import FCN8s

# Initialize model
model = FCN8s(num_classes=21)

# Load and preprocess image (code omitted for brevity)

# Forward pass
output = model(test_input)
```

---

## 📊 Visualization

The project includes visualization code to compare input images, ground truth labels, and model predictions.

## 📜 License

This project is licensed under the MIT License.

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
2. Long, J., Shelhamer, E., & Darrell, T. (2015). *Fully Convolutional Networks for Semantic Segmentation*.
3. VOC2012 Dataset: http://host.robots.ox.ac.uk/pascal/VOC/
