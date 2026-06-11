# Smart Product Price Prediction

A sophisticated **multimodal machine learning solution** for accurate product price prediction using ensemble learning, transfer learning, and advanced feature engineering.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

This project addresses the **Smart Product Pricing Challenge** by developing an intelligent price prediction system that leverages multiple data modalities:

- **Textual Data**: Product descriptions and catalog content
- **Visual Data**: Product images
- **Tabular Features**: Engineered numerical features

The solution employs a **stacked ensemble approach** combining tree-based and deep learning models to achieve robust, high-accuracy price predictions.

---

## Problem Statement

Traditional pricing systems rely on limited data sources. This project tackles the challenge of predicting product prices by:

1. Integrating **multimodal data** (text, images, and structured features)
2. Capturing **non-linear relationships** through ensemble learning
3. Optimizing **blending weights** for superior predictive performance
4. Minimizing **SMAPE** (Symmetric Mean Absolute Percentage Error)

---

## Approach

### **Methodology: Stacked Generalization Ensemble**

Our solution uses a **two-level stacking architecture**:

#### **Level 0: Base Models**
- **LightGBM Regressor**: Captures non-linear patterns in TF-IDF and tabular features
- **Multimodal Neural Network**: Fuses text embeddings, image embeddings, and tabular features through deep learning

#### **Level 1: Meta-Model**
- **Ridge Regressor**: Learns optimal linear combination of base model predictions with L2 regularization

### **Key Preprocessing Steps**

1. **Target Transformation**: Logarithmic transformation (`log1p`) to normalize price distribution
2. **Feature Engineering**: Extracted item pack quantity from product descriptions using regex
3. **Cross-Validation**: 5-Fold CV to prevent data leakage and ensure model robustness

---

## Features

### **Text Features**
- **Item Pack Quantity (IPQ)**: Extracted via regex patterns from product descriptions
- **TF-IDF Vectorization**: Unigram and bigram features capturing keyword importance
- **DistilBERT Embeddings**: 768-dimensional semantic embeddings from `distilbert-base-uncased`

### **Image Features**
- **EfficientNetB0 Transfer Learning**: 1280-dimensional visual feature vectors
- **Image Preprocessing**: Resizing to 224×224 and normalization

### **Tabular Features**
- **Item Quantity**: Count-based feature extracted from catalog content

---

## Model Architecture

### **LightGBM Pipeline**
```
TF-IDF Features + Tabular Features → LightGBM Regressor → OOF Predictions
```

**Hyperparameters:**
- Objective: `regression_l1` (MAE)
- N Estimators: 2000
- Learning Rate: 0.01
- Feature Fraction: 0.8
- Bagging Fraction: 0.8
- L1/L2 Regularization: 0.1

### **Multimodal Neural Network**
```
Tabular Input ─────────────────────────┐
                                       ├─→ Concatenate ─→ Dense(512) ─→ Dense(256) ─→ Output
Text Embeddings ──→ Dense(256) ───────┤                   Dropout      Dropout
                                       │
Image Embeddings ──→ Dense(256) ──────┘
```

**Architecture Details:**
- Modality-specific Dense layers (256 units, ReLU)
- Concatenation layer for feature fusion
- Shared Dense layers with Dropout (0.3) for regularization
- Linear output layer for regression

### **Weight Optimization**
Optimal blending weights are found by minimizing SMAPE on out-of-fold predictions using SLSQP optimization.

---

## Project Structure

```
Code/
├── README.md                              # This file
├── methodology.md.md                      # Detailed methodology documentation
├── data-processing-and-embedddings.ipynb # Data preprocessing & embedding generation
│   ├── Phase 1: Basic preprocessing (log transformation, IPQ extraction)
│   ├── Phase 2: BERT embeddings
│   ├── Phase 3: TF-IDF vectorization
│   ├── Phase 4: Image downloading (parallelized)
│   └── Phase 5: EfficientNetB0 embeddings (parallelized)
│
└── prediction-model.ipynb                # Model training & prediction
    ├── 5-Fold CV pipeline
    ├── LightGBM training
    ├── Neural Network training
    ├── Weight optimization
    └── Submission generation
```

---

## Installation

### **Prerequisites**
- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Kaggle account (if using Kaggle notebooks)

### **Dependencies**

Install required packages:

```bash
pip install pandas numpy scikit-learn lightgbm tensorflow torch transformers pillow tqdm scipy
```

### **Environment Setup**

#### For Google Colab:
```python
!pip install --upgrade transformers huggingface_hub
```

#### For Local Setup:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### **Step 1: Data Processing & Feature Engineering**

Run the data processing notebook to generate embeddings:

```python
# Execute cells in: data-processing-and-embedddings.ipynb
# This will:
# - Load raw datasets
# - Extract item quantities
# - Generate BERT text embeddings
# - Create TF-IDF features
# - Download and embed product images
```

**Expected Outputs:**
- `train_processed.csv` - Processed training data
- `test_processed.csv` - Processed test data
- `train_text_embeddings.npy` - BERT embeddings
- `train_image_embeddings.npy` - EfficientNetB0 embeddings
- `train_tfidf.npz` - TF-IDF sparse matrix

### **Step 2: Train Models & Generate Predictions**

Run the prediction model notebook:

```python
# Execute cells in: prediction-model.ipynb
# This will:
# - Load all preprocessed features
# - Train LightGBM and Neural Network with 5-Fold CV
# - Optimize blending weights
# - Generate final predictions
```

**Output:**
- `submission.csv` - Final price predictions with optimal blending

---

## Results

| Metric | Value |
|--------|-------|
| **CV Strategy** | 5-Fold Cross-Validation |
| **Base Models** | LightGBM + Multimodal NN |
| **Meta-Model** | Ridge Regressor |
| **Evaluation Metric** | SMAPE (Symmetric Mean Absolute Percentage Error) |
| **Target Transform** | Log Transformation (log1p) |

---

## Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
lightgbm>=3.2.0
tensorflow>=2.8.0
torch>=1.9.0
transformers>=4.20.0
Pillow>=8.3.0
tqdm>=4.62.0
scipy>=1.7.0
```

See `requirements.txt` for the complete list with pinned versions.

---

## Key Insights

1. **Model Diversity**: Combining tree-based and neural network models captures complementary patterns
2. **Target Transformation**: Log transformation aligns MSE optimization with SMAPE evaluation metric
3. **Feature Fusion**: Multimodal approach leverages text, image, and structured data effectively
4. **Weight Optimization**: Data-driven blending weights improve ensemble performance over fixed weights
5. **Regularization**: Dropout, L1/L2 penalties, and early stopping prevent overfitting

---

## Contributing

For improvements or bug reports, please create an issue or submit a pull request.

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Contact & Attribution

**Libraries & Models Used:**
- **DistilBERT**: Apache 2.0 License
- **EfficientNetB0**: Apache 2.0 License
- **LightGBM**: MIT License
- **TensorFlow/Keras**: Apache 2.0 License

For detailed methodology, see [methodology.md.md](methodology.md.md)

---

**Last Updated:** 2026-06-11  
**Version:** 1.0
