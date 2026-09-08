# Week 10 — Feature Engineering & Preprocessing

## 🎯 Topics
- **Handling Missing Data**:
  - Detecting null values (`df.isnull().sum()`)
  - Dropping rows / columns (`dropna`)
  - Imputation: Mean, Median (robust to outliers), Mode (categorical)
- **Feature Scaling**:
  - **Standardization (Z-Score)**: $z = \frac{x - \mu}{\sigma}$ (mean = 0, std = 1)
  - **Normalization (Min-Max)**: $x_{\text{norm}} = \frac{x - \min}{\max - \min}$ (range [0, 1])
  - Why algorithms (KNN, Gradient Descent, SVM) fail without scaling
- **Categorical Feature Encoding**:
  - One-Hot Encoding for nominal categories
  - Label / Ordinal Encoding for ranked categories
- **Preventing Data Leakage**: Fitting transformers strictly on the training set

## 🗂️ Structure
```
Week10/
├── README.md
├── notes/
│   └── feature-engineering.md          ← Clean, visual guide to preprocessing
└── mini-project/
    ├── requirements.txt
    ├── dirty_customer_data.csv         ← Sample messy dataset with nulls & raw features
    ├── preprocess.py                   ← End-to-end preprocessing pipeline
    └── cleaned_customer_data.csv       ← Cleaned, imputed, encoded & scaled output
```

## 🚀 Mini-Project: Tabular Preprocessing Pipeline
A clean, practical Python program that:
1. Loads an uncleaned customer dataset with missing values and mismatched scales
2. Detects and visualizes missing data percentages
3. Imputes missing numerical values with the median and missing categories with the mode
4. One-hot encodes categorical text columns (`Department`, `Education`)
5. Standardizes numerical columns (`Age`, `Salary`, `CreditScore`) using `StandardScaler`
6. Exports the final model-ready dataset as `cleaned_customer_data.csv`

## 🔧 Setup & Run
```bash
cd Week10/mini-project
pip install -r requirements.txt
python preprocess.py
```

## ⏱️ Duration: 3 Hours
- **Hour 1**: Missing data strategies (imputation vs deletion)
- **Hour 2**: Feature scaling (StandardScaler vs MinMaxScaler) and One-Hot Encoding
- **Hour 3**: Building and running the complete preprocessing pipeline
