"""
====================================================
Week 10 Mini-Project: Feature Engineering Pipeline
====================================================
Demonstrates:
  1. Missing Data Detection & Analysis
  2. Numerical Imputation (Median) & Categorical Imputation (Mode)
  3. One-Hot Encoding for Categorical Attributes
  4. Feature Standardization (Z-score Scaling)
  5. Exporting Model-Ready Dataset (CSV)
====================================================
Run:    python preprocess.py
Output: Saves 'cleaned_customer_data.csv'
====================================================
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "dirty_customer_data.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "cleaned_customer_data.csv")

def main():
    print("=" * 65)
    print("  WEEK 10: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 65)

    # 1. Load Raw Messy Data
    df = pd.read_csv(INPUT_FILE)
    print(f"\n[1] Raw Dataset Loaded: {len(df)} rows, {len(df.columns)} columns")
    print("-" * 65)
    print(df.head(6))

    # 2. Check Missing Values
    print(f"\n[2] Missing Values Summary:")
    print("-" * 65)
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(1)
    missing_df = pd.DataFrame({"Missing Count": missing_count, "Missing %": missing_pct})
    print(missing_df[missing_df["Missing Count"] > 0])

    # 3. Drop Identifiers
    df_clean = df.drop(columns=["CustomerID"]).copy()

    # 4. Impute Missing Values
    num_cols = ["Age", "AnnualSalary", "CreditScore"]
    cat_cols = ["Department"]

    print(f"\n[3] Handling Missing Data:")
    print(f"    • Numeric cols {num_cols}: Imputing with Median (robust to outliers)")
    num_imputer = SimpleImputer(strategy="median")
    df_clean[num_cols] = num_imputer.fit_transform(df_clean[num_cols])

    print(f"    • Categorical cols {cat_cols}: Imputing with Most Frequent (Mode)")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df_clean[cat_cols] = cat_imputer.fit_transform(df_clean[cat_cols])

    print("    --> Remaining Null Values:", df_clean.isnull().sum().sum())

    # 5. One-Hot Encoding for Categorical Features
    print(f"\n[4] One-Hot Encoding Categorical Columns:")
    df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True, dtype=int)
    print("    --> New Columns Created:", [c for c in df_clean.columns if "Department" in c])

    # 6. Feature Scaling (StandardScaler)
    print(f"\n[5] Feature Scaling with StandardScaler (Z-Score):")
    scaler = StandardScaler()
    
    # Store before stats
    salary_mean_before = df_clean["AnnualSalary"].mean()
    salary_std_before = df_clean["AnnualSalary"].std()

    # Scale numeric columns
    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

    salary_mean_after = df_clean["AnnualSalary"].mean()
    salary_std_after = df_clean["AnnualSalary"].std()

    print(f"    • AnnualSalary BEFORE: Mean = ${salary_mean_before:,.0f}, Std = ${salary_std_before:,.0f}")
    print(f"    • AnnualSalary AFTER:  Mean = {salary_mean_after:.2f}, Std = {salary_std_after:.2f} (Z-Score Normalized)")

    # 7. Final Model-Ready Data
    print(f"\n[6] Cleaned & Preprocessed Model-Ready Dataset:")
    print("-" * 65)
    print(df_clean.head(6).round(3))

    # Export to CSV
    df_clean.round(4).to_csv(OUTPUT_FILE, index=False)
    print(f"\n[7] Export Successful:")
    print(f"    • Saved to: {os.path.basename(OUTPUT_FILE)}")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()
