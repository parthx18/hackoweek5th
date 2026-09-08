# Feature Engineering & Preprocessing — Notes

Real-world raw data is almost never clean. It contains missing fields, unscaled numbers, and categorical text that algorithms cannot directly process.

> *"Garbage In, Garbage Out"* — The quality of your features determines the ceiling of your model's performance.

---

## 1. Handling Missing Data

In pandas, missing values appear as `NaN` (Not a Number) or `None`.

### 1.1 Detection
```python
# Check total nulls per column
print(df.isnull().sum())

# Check percentage of missing values
print(df.isnull().mean() * 100)
```

### 1.2 Strategy 1: Deletion
*   **Drop Rows (`df.dropna()`):**
    *   *When to use:* Missing data is very small (< 2-3% of dataset) and completely random.
    *   *Risk:* Can throw away valuable information if many rows have just 1 missing column.
*   **Drop Columns (`df.drop(columns=[...])`):**
    *   *When to use:* More than 50-60% of the column is missing and cannot be reliably imputed.

### 1.3 Strategy 2: Imputation (Filling In)
Replace missing entries with a representative statistic:

| Imputation Method | Best For | Pros / Cons |
| :--- | :--- | :--- |
| **Mean** | Symmetric / Normal numeric data | Sensitive to extreme outliers |
| **Median** | Skewed numeric data (e.g. Income, Age) | **Recommended**: Robust against outliers |
| **Mode (Most Frequent)** | Categorical data (e.g. City, Gender) | Works on strings and categories |
| **Constant / "Unknown"** | Categorical data | Preserves the fact that data was missing |

```python
from sklearn.impute import SimpleImputer

# Impute numeric columns with median
num_imputer = SimpleImputer(strategy='median')
df['Age'] = num_imputer.fit_transform(df[['Age']])

# Impute categorical columns with most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
df['City'] = cat_imputer.fit_transform(df[['City']])
```

---

## 2. Feature Scaling

Why do we need feature scaling?
Imagine a dataset with two features:
*   **Age:** 20 to 60 (range: 40)
*   **Annual Salary:** \$25,000 to \$150,000 (range: 125,000)

When calculating distance (e.g. in KNN or K-Means) or gradients (in Gradient Descent), Salary will completely dominate Age simply because its numerical numbers are larger, not because it is more important!

### 2.1 Standardization (StandardScaler / Z-Score)
Transforms features so the mean is $0$ and standard deviation is $1$:
$$ z = \frac{x - \mu}{\sigma} $$
*   **When to use:** Most general ML algorithms (Linear/Logistic Regression, SVMs, Neural Networks, PCA).
*   Robust to outliers because it does not bind features to a hard minimum and maximum.

### 2.2 Normalization (MinMaxScaler)
Compresses features strictly into the range $[0, 1]$:
$$ x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} $$
*   **When to use:** Image processing (pixels 0–255 $\to$ 0.0–1.0) or when features have strict known boundaries.
*   Sensitive to extreme outliers (one giant value crushes all other values near 0).

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Age', 'Salary']])
```

---

## 3. Categorical Encoding

Machine learning models only understand numbers, not strings like `"Engineering"`, `"Marketing"`, or `"Sales"`.

### 3.1 One-Hot Encoding (Nominal Data)
Used when categories have **no natural order** (e.g. Red, Blue, Green; Cities; Departments).
Creates a binary dummy column (0 or 1) for each category:

| Department | $\to$ | Dept_Engineering | Dept_Marketing | Dept_Sales |
| :--- | :--- | :---: | :---: | :---: |
| Engineering | | 1 | 0 | 0 |
| Marketing | | 0 | 1 | 0 |
| Sales | | 0 | 0 | 1 |

```python
# In pandas:
df = pd.get_dummies(df, columns=['Department'], drop_first=True)
```
*(Tip: `drop_first=True` prevents the "dummy variable trap" by avoiding collinearity).*

### 3.2 Label / Ordinal Encoding (Ordinal Data)
Used when categories have an **inherent order** (e.g. Low $\to$ 0, Medium $\to$ 1, High $\to$ 2).

---

## 4. The Cardinal Rule: Prevent Data Leakage

**Always split your data into Train and Test BEFORE fitting scalers or imputers!**

```python
# CORRECT WORKFLOW:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit on train, transform on BOTH train and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)      # DO NOT fit on test!
```
If you fit on the test set, information from the test set leaks into the training pipeline, giving an unrealistically optimistic evaluation score.
