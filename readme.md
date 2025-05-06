## Skin Lesion Classification

This is a machine learning project for classifying skin lesions using an unsupervised feature extractor (autoencoder) followed by a classifier. The project is based on PyTorch and uses the Comet ML platform for experiment tracking.

| | |
| --- | --- |
| **Description** | Skin lesion classification using an autoencoder for feature extraction and a classifier for final prediction |
| **Author** | Giovanni Giuseppe Iacuzzo |
| **Course** | [Machine Learning](https://unikore.it) |

---

### Table of Contents

- [Skin Lesion Classification](#skin-lesion-classification-with-autoencoder--classifier)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Code structure](#code-structure)
  - [Usage](#usage)

---

### Introduction

This project performs classification of dermoscopic skin lesion images using a two-step approach:
1. A **Convolutional Autoencoder** is trained to learn compressed image representations (embeddings).
2. A **Feedforward Classifier** is trained on those embeddings to predict the lesion type.

The dataset used in the project contains labeled images of skin lesions, split in a stratified way into training, validation, and test sets.

The entire training and evaluation pipeline is tracked and visualized using [Comet ML](https://www.comet.com/).

The main script is `main.py`, which runs the full pipeline:
- Load and split the dataset
- Train the autoencoder
- Extract and save embeddings
- Train the classifier on embeddings
- Evaluate the classifier on the test set

---

### Requirements

The project runs on **Python 3.11+** and uses the following key libraries:
- `torch` and `torchvision` for model training and data handling
- `matplotlib` and `seaborn` for visualization
- `scikit-learn` for metrics and preprocessing
- `comet_ml` for experiment tracking
- `PyYAML` for configuration handling
- `tqdm` for progress bars

To install the requirements:

```bash
pip install -r requirements.txt
```
### Code structure

```bash
Esame-Machine-Learning/
│
├── dataset/
│   ├── dataset.py               # Dataset loading and stratified splitting
│
├── models/
│   ├── models_autoencoder.py    # Autoencoder architecture
│   ├── model_classifier.py      # Classifier architecture
│
├── utils/
│   ├── utils.py                 # Visualization and utility functions
│   ├── train_autoencoder.py     # Autoencoder training logic
│   ├── train_classifier.py      # Classifier training logic
│   ├── feature_extraction.py    # Embedding extraction using autoencoder
│   ├── test.py                  # Evaluation script
│
├── save_model/                  # Saved embeddings model
├── save_model_autoencoder/      # Saved autoencoder model
├── save_model_classidier/       # Saved classifier model
│
├── config.yml                   # Training and model configuration file
├── requirements.txt             # List of required Python packages
├── .comet.config                # Configuration for comet.ml
│
├── main.py                      # Full pipeline execution script for DNN
│
├── extract_features.py          # Exstract features for all images in dataverse_file
├── vision_embeddings.py         # Extract embeddings from vision models
├── main_XGB_classifier.py       # Model XGB Classifier
│
└── README.md                    # Project documentation
```

### Usage
To reproduce the project, follow these steps:

```bash
git clone [your-repository-url]
cd Esame-Machine-Learning
bash prepare.sh
python main.py
```
You can modify the training parameters and paths in the config.yml file.
