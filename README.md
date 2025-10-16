# Lung and Colon Cancer Detection Project

A deep learning project for detecting and classifying lung and colon cancer from histopathological images using Convolutional Neural Networks (CNN).

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a deep learning model to classify histopathological images into five categories:
- **Lung Adenocarcinoma**
- **Lung Normal**
- **Lung Squamous Cell Carcinoma (SCC)**
- **Colon Adenocarcinoma**
- **Colon Normal**

The project includes:
- A trained CNN model for classification
- A GUI application for easy image prediction
- Jupyter notebooks for training and experimentation

## ✨ Features

- 🔬 Multi-class classification (5 cancer types)
- 🖥️ User-friendly GUI for predictions
- 📊 Comprehensive training notebooks
- 🎯 High accuracy model
- 📈 Data visualization and analysis tools

## 📊 Dataset

The project uses the **Lung and Colon Cancer Histopathological Images** dataset.

**Dataset Structure:**
```
lung_colon_image_set/
├── colon_image_sets/
│   ├── colon_aca/      # Colon Adenocarcinoma
│   └── colon_n/        # Colon Normal
└── lung_image_sets/
    ├── lung_aca/       # Lung Adenocarcinoma
    ├── lung_n/         # Lung Normal
    └── lung_scc/       # Lung Squamous Cell Carcinoma
```

> **Note:** The dataset is NOT included in this repository due to its large size. You need to download it separately.

### Download Dataset:
1. Download from [Kaggle - LC25000 Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
2. Extract to project root as `lung_colon_image_set/`

## 🔧 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- GPU support (optional, for training)

## 📥 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/lung-colon-cancer-detection.git
cd lung-colon-cancer-detection
```

### 2. Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install tensorflow keras numpy pandas opencv-python pillow matplotlib seaborn scikit-learn
```

### 4. Download Pre-trained Model

The pre-trained model (`Model.h5`) is stored using Git LFS. After cloning:
```bash
git lfs pull
```

If Git LFS is not installed:
- **Windows:** Download from [Git LFS](https://git-lfs.github.com/)
- **macOS:** `brew install git-lfs`
- **Linux:** `sudo apt-get install git-lfs`

Then run: `git lfs install` and `git lfs pull`

### 5. Download Dataset (Required for Training)

Download the dataset and place it in the project root:
```
lung-colon-cancer-detection/
├── lung_colon_image_set/
│   ├── colon_image_sets/
│   └── lung_image_sets/
├── Model.h5
├── gui.py
└── ...
```

## 🚀 Usage

### Option 1: GUI Application (Recommended for Predictions)

1. **Update Model Path** in `gui.py` (line 9):
   ```python
   # Change this line to your actual path:
   loaded_model = tf.keras.models.load_model("Model.h5", compile=False)
   ```

2. **Run the GUI:**
   ```bash
   python gui.py
   ```

3. **Use the Application:**
   - Click "Upload Image" to select a histopathological image
   - The model will predict and display the cancer type
   - View probability scores for each class

### Option 2: Using Jupyter Notebooks

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open Notebooks:**
   - `mlproject.ipynb` - Main training and evaluation notebook
   - `backend.ipynb` - Backend processing and model training

3. **Run cells sequentially** to train or evaluate the model

### Option 3: Training from Scratch

1. **Ensure dataset is downloaded** to `lung_colon_image_set/`

2. **Run the training script:**
   ```bash
   python mlp.py
   ```
   
   Or use the Jupyter notebooks for interactive training.

## 📁 Project Structure

```
lung-colon-cancer-detection/
│
├── lung_colon_image_set/        # Dataset (not in repo, download separately)
│   ├── colon_image_sets/
│   └── lung_image_sets/
│
├── Model.h5                      # Trained model (Git LFS)
├── mlp.py                        # Model training script
├── gui.py                        # GUI application
├── a.py                          # Additional utilities
├── mlproject.ipynb               # Training notebook
├── backend.ipynb                 # Backend notebook
├── .gitignore                    # Git ignore file
├── .gitattributes                # Git LFS configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🧠 Model Information

### Architecture
- **Type:** Convolutional Neural Network (CNN)
- **Input Shape:** 224x224x3 (RGB images)
- **Output:** 5 classes (softmax activation)
- **Framework:** TensorFlow/Keras

### Training Details
- **Optimizer:** Adamax
- **Loss Function:** Categorical Crossentropy
- **Data Augmentation:** Yes (rotation, flip, zoom)
- **Image Size:** 224x224 pixels

### Performance
- High accuracy on test data
- Confusion matrix and classification reports available in notebooks

## 📊 Results

The model achieves high accuracy in classifying the five cancer types. Detailed results including:
- Confusion matrices
- Classification reports
- Training/validation curves
- Sample predictions

Can be found in the Jupyter notebooks.

## 🔍 Troubleshooting

### Common Issues:

1. **"Model.h5 not found" error:**
   - Ensure Git LFS is installed: `git lfs install`
   - Pull LFS files: `git lfs pull`
   - Or download model separately if needed

2. **"No module named 'tensorflow'" error:**
   - Install TensorFlow: `pip install tensorflow`
   - Ensure virtual environment is activated

3. **GUI doesn't open:**
   - Install tkinter: 
     - Windows: Included with Python
     - Linux: `sudo apt-get install python3-tk`
     - macOS: Included with Python

4. **Path errors in gui.py:**
   - Update the model path on line 9 to match your directory structure
   - Use absolute or relative paths correctly

5. **Out of memory errors during training:**
   - Reduce batch size in training scripts
   - Close other applications
   - Use GPU if available

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

Your Name - [Your GitHub Profile]

## 🙏 Acknowledgments

- Dataset: [LC25000 Lung and Colon Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- TensorFlow/Keras teams
- Open source community

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note:** This project is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.
