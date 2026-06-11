# 🧠 Smart Product Price Prediction – Methodology

## Overview
Our solution addresses the **Smart Product Pricing Challenge** by employing a **multimodal, stacked ensemble learning approach**.  
This method holistically analyzes product information by integrating **textual data, visual data, and engineered features** to produce a robust and accurate price prediction.

---

## ⚙️ Methodology Used

The core of our methodology is a **stacked generalization (stacking) ensemble**.  
This technique combines the predictions of multiple diverse base models (**Level 0**) and uses them as input features for a final **meta-model (Level 1)** that learns the optimal way to blend their outputs.  
This approach leverages the unique strengths of different algorithms, leading to a more accurate and generalized final prediction than any single model could achieve.

### 🧩 Target Transformation
A critical preprocessing step was the **logarithmic transformation** of the target variable `price` using `numpy.log1p`.  
Product prices typically have a right-skewed distribution; this transformation:
- Normalizes the target variable.
- Improves regression model performance.
- Aligns the training objective (MSE) with the competition’s relative evaluation metric, **SMAPE**.

All predictions were **inverse-transformed** using `numpy.expm1` before submission.

### 🔁 Cross-Validation
A **5-Fold Cross-Validation** strategy was used to:
- Train the base models.
- Generate **out-of-fold predictions** (training data for the meta-model).
- Prevent data leakage and ensure model robustness.

---

## 🧱 Model Architecture / Algorithms Selected

Our ensemble was constructed with a deliberate focus on **model diversity** across both architecture and feature representation.

### **Base Models (Level 0)**

#### 1. LightGBM Model
A **gradient-boosted decision tree** trained on:
- Engineered **tabular features**.
- High-dimensional, sparse **TF-IDF text features**.

LightGBM excels at capturing **non-linear interactions** and **thresholds** in tabular and sparse data.

#### 2. Multimodal Neural Network
A **deep learning model** with an **intermediate fusion architecture** integrating multiple data modalities.

**Architecture:**
- Separate input branches for:
  - Tabular features  
  - Text embeddings (from **DistilBERT**)  
  - Image embeddings (from **EfficientNetB0**)
- Modality-specific Dense layers to process and align high-dimensional embeddings.
- **Concatenate layer** to fuse all features.
- Shared Dense layers with **Dropout** for regularization and complex pattern learning.

---

### **Meta-Model (Level 1)**

#### Ridge Regressor
A simple yet robust **linear model** used as the meta-model.  
It learns the **optimal linear combination** of the Level 0 predictions while applying **L2 regularization** to prevent overfitting.

---

## 🧮 Feature Engineering Techniques Applied

A multi-pronged **feature engineering strategy** was employed to extract maximum value from the raw data.

### **Textual Features (`catalog_content`)**

#### Parsing
- Extracted a numerical feature: **Item Pack Quantity (IPQ)** using **regular expressions**.
- Default value of `1` imputed where IPQ was missing.

#### TF-IDF Vectorization
- Applied `TfidfVectorizer` to generate unigram and bigram features.
- Captures the importance of keywords, model numbers, and technical terms correlated with price.

#### Transformer Embeddings
- Used **`distilbert-base-uncased`** (Apache 2.0 license).
- Generated **768-dimensional embeddings** capturing **semantic context** of product descriptions.

---

### **Visual Features (`image_link`)**

#### Transfer Learning
- Employed **transfer learning** using a **pre-trained CNN** as a fixed feature extractor.

#### Model Selection
- **EfficientNetB0** chosen for its **state-of-the-art accuracy** and **efficiency** (Apache 2.0 license).

#### Process
1. Downloaded and resized images to **224x224**.  
2. Normalized pixel values.  
3. Passed through EfficientNetB0 (final classification layer removed).  
4. Extracted **1280-dimensional dense feature vectors** representing rich visual product information.

---

## 🧩 Conclusion

The success of this approach lies in the **principle of diversity**.  
By combining:
- A **tree-based model** (LightGBM) effective with sparse, keyword-driven features, and  
- A **neural network** that captures deep semantic and visual context,  

the stacked ensemble achieves **robust and informed price predictions**.

Additional strengths:
- **Target normalization** enhanced model stability.
- **Modular, resource-aware pipeline** enabled efficient multimodal data processing.

---

**Final Takeaway:**  
> A fusion of structured learning, deep multimodal understanding, and ensemble optimization — delivering a smarter, data-driven product pricing solution.
