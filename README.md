# Multi-Modal Bone Fracture Detection
A multi-modal deep learning project for bone fracture detection using X-ray images. This project implements and compares three distinct models—YOLOv8, ResNet, and a custom CNN—to classify fractured versus non-fractured bones.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Models](#models)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
Bone fractures require accurate, timely diagnosis. This project leverages the strengths of multiple deep learning architectures to automatically detect fractures from X-ray images. By implementing a YOLOv8-based model, a ResNet-based model, and a custom CNN, we provide a comprehensive comparison in performance—evaluated on metrics such as accuracy, precision, recall, and F1-score.
##  Project Structure

```sh
└── Multi-model-BoneFracture-Detection.git/
    ├── InferenceWeights
    │   └── .gitkeep
    ├── Streamlit Front End
    │   ├── StreamlitBoneapp.ipynb
    │   ├── app.py
    │   ├── app1.py
    │   └── sidebar.py
    ├── data-set
    │   ├── test
    │   ├── train
    │   └── val
    ├── histories
    │   └── cnn_fine_tuned_history.pkl
    ├── images
    │   ├── NN_Architecture.jpg
    │   ├── NN_Architecture_Updated.jpg
    │   └── medical.jpg
    ├── training
    │   ├── CustomCNN.ipynb
    │   ├── ResNet.ipynb
    │   └── yolov8.ipynb
    ├── weights
    │   └── .gitkeep
    └── yolo
        ├── runs
        └── yoloDataSet
```
## Dataset
The dataset, sourced from [Kaggle](https://www.kaggle.com/), consists of X-ray images labeled for the presence or absence of bone fractures. The data is partitioned into training, validation, and test sets.

## Requirements
The project is built in Python and requires the following libraries:
- **TensorFlow**
- **Keras**
- **Pillow (PIL)**
- **NumPy**
- **Matplotlib**

*Optional:* [Google Colab](https://colab.research.google.com/) can be used for cloud-based model training.

## Models

### 1. YOLOv8-based Model
- **Objective:** Object detection and binary classification (fractured vs. non-fractured).
- **Highlights:**  
  - Fine-tuned for localizing fracture regions within X-ray images.
  - Uses the Adam optimizer and Binary Crossentropy loss.
  
### 2. ResNet-based Model
- **Objective:** Leverage deep residual networks for robust feature extraction.
- **Highlights:**  
  - Incorporates skip connections to improve training in deeper layers.
  - Custom fully connected layers are added for binary classification.
  - Uses the Adam optimizer and Binary Crossentropy loss.

### 3. Custom CNN Model
- **Objective:** A traditional CNN built from scratch for fracture detection.
- **Highlights:**  
  - Contains multiple convolutional, pooling, and fully connected layers.
  - Uses the Adam optimizer and Binary Crossentropy loss.

## Data Preparation
Effective data preprocessing is critical. The pipeline includes:

- **Image Augmentation:**  
  Using Keras’ `ImageDataGenerator`, images are augmented through random rotations, flips, zooming, and shifts.
  
- **Preprocessing:**  
  All images are resized and normalized to ensure consistency prior to training.
## Training and Evaluation
- **Data Split:** The dataset is divided into training, validation, and test subsets.
- **Training:**  
  - Each model is trained using similar hyperparameters such as batch size, number of epochs, and learning rate.
  - Hyperparameter tuning is conducted to optimize performance.
- **Evaluation Metrics:**  
  Accuracy, loss, precision, recall, and F1-score are computed.
- **Visualization:**  
  Confusion matrices and performance plots are generated for in-depth error analysis.

## Results
Each model’s performance is compared to determine the most effective approach for detecting bone fractures. Detailed metrics and visualizations provide insights into the strengths and weaknesses of each architecture.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/hamzabhat/Multi-model-BoneFracture-Detection.git
pip install -r requirements.txt
```
## Usage
### Training the Models
Run the training scripts for each model:
```bash
python training/yolov8.ipynb
python training/ResNet.ipynb
python training/CustomCNN.ipynb
```

### Evaluation
After training, execute the evaluation script to generate performance reports:
```bash
python evaluate_models.py
```
## Contributing
Contributions are welcome! To contribute, follow these steps:  

1. **Fork the repository** on GitHub.  
2. **Clone your fork** locally:  
   ```bash
   git clone https://github.com/your-username/Multi-model-BoneFracture-Detection.git
   ```

3. **Create a new branch** for your feature or fix:
```
git checkout -b feature-branch
```
4. **Make changes and commit them**:
```
git commit -m "Added a new feature"
```
5. **Push to your fork** and **create a pull request**:
```
git push origin feature-branch
```
4. **Submit a pull request** and describe your changes.

#### Feel free to open an issue for any improvements or questions!

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for further details

