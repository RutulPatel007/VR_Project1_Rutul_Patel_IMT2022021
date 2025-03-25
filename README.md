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

- **Source**: The dataset was sourced from the following repositories:
  - [Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
  - [Masked Face Segmentation Dataset](https://github.com/sadjadrz/MFSD)
- **Structure**:
  - **Image Classification**:
    - Two categories: `with_mask` and `without_mask`, organized into separate folders.
    - Image sizes varied, standardized to 64x64.
    - Dataset was divided into 80:20 for training and testing
  - **Segmentation**:
    - Two folders: `crop_face` containing cropped face images and `segmented_mask` containing corresponding segmentation masks.
    - Image sizes standardized to 256x256.

Note - We only uploaded the classification dataset to github repo and not segmentation dataset since it was very large and we used gdown library in colab to download


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
- **Data Preprocessing**:
  - Image Normalization: Scaled pixel values to the range [0, 1] by dividing by 255.
- **Data Augmentation**:
  - Applied rotation, zoom, shift, and horizontal flip using `ImageDataGenerator`.
- **Hyperparameters (Giving best results)**:
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Batch Size: 32
  - Epochs: 50

### c. Region Segmentation Using Traditional Techniques

- **Method1**:
  - Applied both Thresholding and edge detection techniques,used Dataset.csv as Ground truth values
- **Method2**:
  - Applied both Thresholding and edge detection techniques,used Dataset.csv as Ground truth values

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

## Hyperparameters and Experiments

### 🔬 CNN Model Experiments

#### **1. Initial Model Setup**
- **Architecture**: Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output.
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 50

#### **2. Experiment Variations and Results**

| Experiment | Learning Rate | Optimizer | Batch Size | Accuracy (%) |
|------------|--------------|-----------|------------|--------------|
| Baseline   | 0.001        | Adam      | 32         | 97.1         |
| Exp 1      | 0.0001       | Adam      | 32         | 96.2         |
| Exp 2      | 0.001        | SGD       | 32         | 94.5         |
| Exp 3      | 0.001        | Adam      | 64         | 96.8         |
| Exp 4      | 0.0005       | RMSprop   | 32         | 96.9         |

- **Observations:**
  - Adam optimizer with a learning rate of `0.001` performed the best.
  - SGD showed lower accuracy due to slower convergence.
  - Increasing batch size slightly reduced accuracy, possibly due to less frequent weight updates.
  - Reducing the learning rate (`0.0001`) led to slower training but maintained decent accuracy.

---

### 🧪 U-Net Model Experiments

#### **1. Initial Model Setup**
- **Architecture**: Encoder-decoder with skip connections.
- **Loss Function**: Binary Crossentropy + Dice Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 16
- **Dropout Rate**: 0.3
- **Epochs**: 30

#### **2. Experiment Variations and Results**

| Experiment | Learning Rate | Batch Size | Dropout Rate | IoU Score | Dice Score |
|------------|--------------|------------|--------------|-----------|------------|
| Baseline   | 0.001        | 16         | 0.3          | 0.91      | 0.94       |
| Exp 1      | 0.0001       | 16         | 0.3          | 0.87      | 0.90       |
| Exp 2      | 0.001        | 32         | 0.3          | 0.89      | 0.92       |
| Exp 3      | 0.001        | 16         | 0.5          | 0.88      | 0.91       |
| Exp 4      | 0.0005       | 16         | 0.3          | 0.90      | 0.93       |

- **Observations:**
  - The optimal learning rate was found to be `0.001`, with `0.0001` leading to slower convergence.
  - Increasing batch size to 32 led to a minor decrease in IoU and Dice scores, indicating smaller batch sizes are preferable.
  - Higher dropout (`0.5`) reduced performance slightly, suggesting `0.3` is a good balance for regularization.
  - The Adam optimizer at `0.001` worked best, similar to the CNN classification experiments.

---

### 📌 Final Takeaways
- **For CNN**, Adam optimizer with `0.001` learning rate and batch size `32` provided the highest accuracy.
- **For U-Net**, `0.001` learning rate, batch size `16`, and dropout rate `0.3` yielded the best segmentation results.




## 📊 Results

- Note: For classification, data was divided into 80-20 for testing and training.

| Task                                | Model           | Accuracy (%) | IoU Score | Dice Score |
|-------------------------------------|-----------------|--------------|-----------|------------|
| Binary Classification (ML)         | SVM             | 91.2         | -         | -          |
|                                     | MLP             | 93.5         | -         | -          |
| Binary Classification (CNN)        | CNN             | **97.8**     | -         | -          |
| Region Segmentation (Traditional Method1)  | Edge Detection  | -            | 0.09      | 0.16       |
| Region Segmentation (Traditional Method2)  | Edge Detection  | -            | 0.10      | 0.18       |
| Region Segmentation (Traditional Method1)  | Thresholding  | -            | 0.83      | 0.89       |
| Region Segmentation (Traditional Method2)  | Thresholding  | -            | 0.29      | 0.42       |
| Mask Segmentation (Deep Learning)  | U-Net           | -            | **0.91**  | **0.94**   |


- **CNN vs ML Classifiers**: CNN outperformed traditional classifiers with significant margin. Also MLP outperformed SVM 
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

### Note
- The project was originally developed using Jupyter Notebook (`.ipynb`) files, except for the C part which was done in .py.
- Both `.py` and `.ipynb` versions of the files are available.
- Running in Google Colab is preferred for Jupyter Notebook execution.
- Clone the repo and go to the root folder of repo

### 1. Environment Setup
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Steps to Run
#### (a) Handcrafted Features + CNN Model (Part A & B)
- This step includes both feature extraction, ML classifiers, and CNN model training.
- Run using:
  - Python script:
    ```bash
    python classify_ml.py
    ```
  - Jupyter Notebook (Colab preferred):
    ```bash
    jupyter notebook classify_ml.ipynb
    ```

#### (b) Traditional Segmentation (PART C)
- Run edge detection and thresholding:
  - Python script:
    ```bash
    python traditional_segmentation.py
    ```


#### (c) U-Net Model (PART D)
- Train U-Net:
  - Python script:
    ```bash
    python train_unet.py
    ```
  - Jupyter Notebook:
    ```bash
    jupyter notebook train_unet.ipynb
    ```


## 📁 Project Structure

```
├── PartA&B                       # Classifications code
|   |── Classification.ipynb      # Contains code for both Handcrafted and CNN classification
|   |── Classification.py         # .py version 
├── requirements.txt              # Dependencies
├── README.md                     # Project report
├── data/
│   ├── Face_Mask_Detection/      # Dataset folder for classification
│   │   ├── with_mask/            # Images of people wearing masks
│   │   ├── without_mask/         # Images of people without masks
├── MSFD/
│   ├── 1/      
│   │   ├── dataset.csv/                  # The groundtruth values used in Method1
│   │   ├── img/                          # The original images
│   │   ├── face_crop/                    # The cropped images of original images
│   │   ├── face_crop_segmentation/       # The segmented output of cropped images
│   ├── 2/                                # The groundtruth values used in Method2
│   │   ├── img/                          #sample images
├── PART C/
│   ├── METHOD 1/                 # This method uses Dataset.csv as ground truth values
│   │   ├── segmented_output/     # The output images aswell the accuracy 
│   │   ├── Preprocess.py/        # Script for making same file names 
│   │   ├── result.py/            # Genrate the output and accuracy
│   ├── METHOD 2/                 # This method uses face_crop_segmentation as ground truth values
│   │   ├── segmented_output/     # The output images aswell the accuracy 
│   │   ├── Preprocess.py/        # Script for making same file names 
│   │   ├── result.py/            # Genrate the output and accuracy
```

---


## 🎓 Team Members

- **Rutul Patel** (IMT2022021) - [GitHub](https://github.com/rutulpatel)
- **Aryaman Pathak** (IMT2022513) - [GitHub](https://github.com/aryamanpathak2022)
- **Shreyas Biradar** (IMT2022529) - [GitHub](https://github.com/BiradarScripts)
