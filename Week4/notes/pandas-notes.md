# Pandas — Notes

## What is Pandas?
Pandas is the go-to Python library for data manipulation and analysis.  
Core objects: **Series** (1D) and **DataFrame** (2D table).

```bash
pip install pandas
```

---

## 1. Creating DataFrames

```python
import pandas as pd

# From a dict
df = pd.DataFrame({
    'name':   ['Alice', 'Bob', 'Charlie'],
    'grade':  [88,      75,    92],
    'course': ['CS',    'IT',  'CS'],
})

# From a CSV file
df = pd.read_csv('students.csv')

# From Excel
df = pd.read_excel('data.xlsx')
```

---

## 2. Exploring a DataFrame

```python
df.head()        # first 5 rows
df.tail(3)       # last 3 rows
df.shape         # (rows, cols) → (3, 3)
df.columns       # Index of column names
df.dtypes        # data type of each column
df.info()        # non-null counts + dtypes
df.describe()    # count, mean, std, min, max for numeric cols
df.isnull().sum()# count of missing values per column
```

---

## 3. Selecting Data

```python
# Select a column (returns Series)
df['name']
df.name          # same, dot notation

# Select multiple columns (returns DataFrame)
df[['name', 'grade']]

# Select rows by index
df.iloc[0]       # first row
df.iloc[0:3]     # rows 0,1,2

# Select rows by label
df.loc[0]        # row with index label 0

# Filter rows by condition
df[df['grade'] >= 80]                           # grade ≥ 80
df[(df['grade'] >= 80) & (df['course'] == 'CS')] # AND condition
df[df['name'].isin(['Alice', 'Bob'])]            # in a list
```

---

## 4. Data Cleaning

```python
# Check for missing values
df.isnull().sum()

# Drop rows with ANY null
df.dropna()

# Drop rows only if ALL values are null
df.dropna(how='all')

# Fill missing values
df['grade'].fillna(0)           # fill with 0
df['grade'].fillna(df['grade'].mean())  # fill with mean

# Drop duplicate rows
df.drop_duplicates()

# Rename columns
df.rename(columns={'name': 'student_name', 'grade': 'marks'}, inplace=True)

# Change data type
df['grade'] = df['grade'].astype(float)

# Strip whitespace from string columns
df['name'] = df['name'].str.strip()

# Replace values
df['course'].replace({'CS': 'Computer Science', 'IT': 'Inf. Tech'}, inplace=True)
```

---

## 5. Adding & Modifying Columns

```python
# Add a new column
df['passed'] = df['grade'] >= 60

# Derived column
df['grade_letter'] = df['grade'].apply(
    lambda g: 'A' if g >= 90 else ('B' if g >= 80 else 'C')
)

# Apply a function to a column
def categorize(score):
    if score >= 90: return 'Excellent'
    elif score >= 75: return 'Good'
    else: return 'Needs Improvement'

df['category'] = df['grade'].apply(categorize)
```

---

## 6. Merging / Joining DataFrames

```python
students = pd.DataFrame({'id': [1,2,3], 'name': ['Alice','Bob','Charlie']})
grades   = pd.DataFrame({'id': [1,2,4], 'grade': [88, 75, 95]})

# Inner join — only matching rows
pd.merge(students, grades, on='id', how='inner')

# Left join — all from left, matched from right
pd.merge(students, grades, on='id', how='left')

# Outer join — all rows from both
pd.merge(students, grades, on='id', how='outer')

# Concatenate vertically (stack rows)
pd.concat([df1, df2], ignore_index=True)
```

---

## 7. GroupBy

GroupBy = split data into groups → apply aggregation → combine results.

```python
# Average grade per course
df.groupby('course')['grade'].mean()

# Multiple aggregations
df.groupby('course').agg(
    avg_grade  = ('grade', 'mean'),
    max_grade  = ('grade', 'max'),
    count      = ('name',  'count'),
)

# GroupBy + filter
df.groupby('course').filter(lambda g: g['grade'].mean() >= 80)

# Pivot table (like Excel pivot)
pd.pivot_table(df, values='grade', index='course', aggfunc='mean')
```

---

## 8. Sorting & Ranking

```python
df.sort_values('grade', ascending=False)         # sort by grade desc
df.sort_values(['course', 'grade'])              # multi-column sort
df['rank'] = df['grade'].rank(ascending=False)   # add rank column
```

---

## Quick Reference

| Task | Code |
|------|------|
| Read CSV | `pd.read_csv('file.csv')` |
| Shape | `df.shape` |
| Filter | `df[df['col'] > val]` |
| Group mean | `df.groupby('col')['val'].mean()` |
| Drop nulls | `df.dropna()` |
| Fill nulls | `df.fillna(0)` |
| Merge | `pd.merge(df1, df2, on='id')` |
| Sort | `df.sort_values('col')` |
| Apply fn | `df['col'].apply(func)` |
