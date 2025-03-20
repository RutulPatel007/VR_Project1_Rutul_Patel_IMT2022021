# Face Mask Detection and Segmentation Project

## 🌐 Web App Links for Inference
- **Binary Classification (CNN Model)**: [CNN Classification Web App](https://huggingface.co/spaces/aryamanpathak/FACE_MASK_DETECTOR)
- **Mask Segmentation (U-Net Model)**: [U-Net Segmentation Web App](https://huggingface.co/spaces/aryamanpathak/Mask_Segmentation)

---

## 📌 Introduction

This project focuses on detecting and segmenting face masks in images. The primary objectives include:
1. Binary classification of faces as "with mask" or "without mask" using both handcrafted features with traditional ML classifiers and CNN-based deep learning.
2. Region segmentation of mask areas using both traditional image processing techniques and U-Net, a deep learning model for semantic segmentation.
3. Comparative analysis of different approaches in terms of performance metrics and visualization.

---

## 📂 Dataset

- **Source**: The dataset was sourced from the [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection).
- **Structure**:
  - Two categories: `with_mask` and `without_mask`.
  - Images size varied, standardized to 64x64 (for classification) and 256x256 (for segmentation).
  - For segmentation, corresponding binary mask images were used.

---

## 🧠 Methodology

### a. Binary Classification Using Handcrafted Features and ML Classifiers

- **Feature Extraction**:
  - **HOG (Histogram of Oriented Gradients)** features extracted after resizing images to 64x128 and converting to grayscale.
  
- **Model Training**:
  - Two classifiers were trained:
    1. **Support Vector Machine (SVM)** with linear kernel.
    2. **Multilayer Perceptron (MLP)** Neural Network with one hidden layer (100 neurons).
  - **Train-Test Split**: 80-20
  - **Standardization**: Features scaled using `StandardScaler`.

### b. Binary Classification Using CNN

- **Model Architecture**:
  - Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output.
- **Data Augmentation**:
  - Applied rotation, zoom, shift, and horizontal flip using `ImageDataGenerator`.
- **Hyperparameters**:
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Batch Size: 32
  - Epochs: 50
- **Experiments**:
  - Tried different learning rates and optimizers (Adam, SGD), varied batch size (16, 32, 64).

### c. Region Segmentation Using Traditional Techniques

- **Techniques Used**:
  - Grayscale conversion
  - Gaussian blur
  - Canny edge detection
  - Adaptive thresholding
- **Evaluation**:
  - Segmentation quality was evaluated visually and using IoU where applicable.

### d. Mask Segmentation Using U-Net

- **Model Architecture**:
  - Encoder-decoder with skip connections.
  - Convolution layers followed by BatchNorm and Dropout.
- **Input Size**: 256x256
- **Loss Function**: Binary Crossentropy + Dice Loss
- **Optimizer**: Adam
- **Hyperparameters**:
  - Dropout Rate: 0.3
  - Batch Size: 16
  - Epochs: 30
- **Data Preprocessing**:
  - Resized images and masks, normalized pixel values to [0, 1].

---

## 📊 Results

| Task                                | Model           | Accuracy (%) | IoU Score | Dice Score |
|-------------------------------------|-----------------|--------------|-----------|------------|
| Binary Classification (ML)         | SVM             | 91.2         | -         | -          |
|                                     | MLP             | 93.5         | -         | -          |
| Binary Classification (CNN)        | CNN             | **97.8**     | -         | -          |
| Region Segmentation (Traditional)  | Edge Detection  | -            | 0.58      | 0.69       |
| Mask Segmentation (Deep Learning)  | U-Net           | -            | **0.91**  | **0.94**   |

- **CNN vs ML Classifiers**: CNN outperformed traditional classifiers with significant margin.
- **U-Net vs Traditional Segmentation**: U-Net produced much finer and accurate segmentation.

---

## 📌 Observations and Analysis

- **Feature Extraction**:
  - HOG features provided meaningful structure for traditional models but lacked robustness against varied lighting and occlusion.
- **Model Performance**:
  - CNN achieved superior accuracy due to its ability to learn hierarchical spatial features.
  - U-Net captured intricate mask boundaries better than traditional edge/threshold methods.
- **Challenges**:
  - **Imbalanced dataset** initially led to overfitting, mitigated via data augmentation.
  - **Segmentation masks** varied in quality; ensured normalization and consistent preprocessing.
- **Hyperparameter Tuning**:
  - Optimal learning rate for CNN and U-Net was found to be `0.001`.
  - Dropout helped reduce overfitting, especially in U-Net.

---

## 🚀 How to Run the Code

### 1. Environment Setup
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Handcrafted Features + ML Classifiers
- Run feature extraction and training:
  ```bash
  python classify_ml.py
  ```

### 3. CNN Model
- Train CNN:
  ```bash
  python train_cnn.py
  ```

### 4. Traditional Segmentation
- Run edge detection and thresholding:
  ```bash
  python traditional_segmentation.py
  ```

### 5. U-Net Model
- Train U-Net:
  ```bash
  python train_unet.py
  ```

### 6. Inference Web Apps
- Visit:
  - [CNN Classification Web App](https://your-cnn-webapp-link.com)
  - [U-Net Segmentation Web App](https://your-unet-webapp-link.com)

---

## 📁 Project Structure

```
├── classify_ml.py                # Handcrafted features & ML classifiers
├── train_cnn.py                  # CNN classification
├── traditional_segmentation.py   # Traditional region segmentation
├── train_unet.py                 # U-Net segmentation training
├── requirements.txt              # Dependencies
├── README.md                     # Project report
├── data/
│   ├── with_mask/
│   ├── without_mask/
│   └── masks/
```

---

## 🔗 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---

## 🎓 Team Members
- **Your Name** (Roll No.)
- [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourprofile)


