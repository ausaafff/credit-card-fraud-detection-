# Credit Card Fraud Detection

## Overview

This project detects fraudulent credit card transactions using machine learning on the Kaggle Credit Card Fraud Detection dataset (284,807 transactions, 0.17% fraud). It includes data preprocessing, EDA, feature selection, and model evaluation with Logistic Regression, Decision Tree, and Random Forest, addressing class imbalance via SMOTE.

---

## Dataset

- **Source:** `creditcard.csv` (Kaggle)  
- **Size:** 284,807 rows (275,663 after deduplication), 31 columns  
- **Features:** Time, V1–V28 (PCA-transformed), Amount, Class (0 = non-fraud, 1 = fraud)  
- **Imbalance:** 99.83% non-fraud, 0.17% fraud  

---

## Project Structure

- `fraud_detection.ipynb`: Full pipeline (preprocessing, EDA, modeling)  
- `creditcard.csv`: Dataset (download from Kaggle)  
- `README.md`: This file  

---

## Requirements

- Python 3.9+
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `imblearn`

---

## Usage

1. **Download Dataset:**  
   Place `creditcard.csv` in the project directory.

2. **Run the Notebook:**  
   Open and execute `fraud_detection.ipynb` in Jupyter Notebook.

---

## Methodology

### Preprocessing:
- Standardize `Amount`
- Drop `Time`
- Remove duplicates

### EDA:
- Class distribution plot  
- Correlation analysis  
- PCA visualization  

### Feature Selection:
- Select features with Random Forest importance ≥ 0.02 (~10 features)

### Data Splitting:
- 70% train, 20% validation, 30% test (stratified)

### Modeling:
- **Models:** Logistic Regression, Decision Tree, Random Forest  
- **Evaluate:** Pre- and post-SMOTE  
- **Metrics:** Accuracy, Precision, Recall, F1, AUC  

---

## Results

### Pre-SMOTE (Test Set, 142 Frauds):

| Model               | Precision | Recall | F1   | AUC   |
|--------------------|-----------|--------|------|-------|
| Random Forest       | 0.93      | 0.72   | 0.81 | 0.928 |
| Logistic Regression | 0.85      | 0.54   | 0.66 | 0.962 |
| Decision Tree       | 0.74      | 0.70   | 0.72 | 0.848 |

### Post-SMOTE:

| Model               | Precision | Recall | F1   | AUC   |
|--------------------|-----------|--------|------|-------|
| Random Forest       | 0.74      | 0.78   | 0.76 | 0.954 |
| Logistic Regression | 0.06      | 0.88   | 0.11 | 0.967 |
| Decision Tree       | 0.37      | 0.74   | 0.49 | 0.869 |

> **Key Insight:** Random Forest post-SMOTE detects 78% of frauds with strong performance.

---

