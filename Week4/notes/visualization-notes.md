# Data Visualization — Matplotlib & Seaborn Notes

## Libraries Overview

| Library | Best For | Style |
|---------|----------|-------|
| **Matplotlib** | Full control, any chart type | Manual/verbose |
| **Seaborn** | Statistical plots, beautiful defaults | High-level / easy |

```bash
pip install matplotlib seaborn
```

---

## Matplotlib Basics

```python
import matplotlib.pyplot as plt

# Basic flow:
# 1. Create figure + axes
# 2. Plot data
# 3. Add labels/title
# 4. Show or save

fig, ax = plt.subplots(figsize=(8, 5))  # width x height in inches
ax.plot([1,2,3,4], [10,20,15,25])       # line chart
ax.set_title("My Chart")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
plt.tight_layout()
plt.savefig('chart.png', dpi=150)       # save to file
plt.show()
```

---

## Common Matplotlib Chart Types

```python
# Line chart
ax.plot(x, y, color='blue', linewidth=2, linestyle='--', label='Sales')

# Bar chart
ax.bar(categories, values, color='steelblue', edgecolor='white')

# Horizontal bar
ax.barh(categories, values)

# Scatter plot
ax.scatter(x, y, color='red', s=100, alpha=0.7)  # s = marker size

# Histogram
ax.hist(data, bins=10, color='green', edgecolor='black')

# Pie chart
ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

# Multiple plots in one figure (subplots)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 cols
axes[0].plot(x, y)
axes[1].bar(cats, vals)
```

---

## Seaborn Basics

Seaborn works directly with Pandas DataFrames — much simpler!

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set theme (optional, makes it beautiful)
sns.set_theme(style="darkgrid")  # whitegrid, dark, white, ticks
```

---

## Common Seaborn Chart Types

```python
# Distribution plot (histogram + KDE curve)
sns.histplot(df['grade'], bins=10, kde=True)

# Box plot — median, quartiles, outliers
sns.boxplot(x='course', y='grade', data=df)

# Violin plot — like boxplot + distribution shape
sns.violinplot(x='course', y='grade', data=df)

# Bar plot with error bars
sns.barplot(x='course', y='grade', data=df, estimator='mean')

# Scatter plot with regression line
sns.regplot(x='attendance', y='grade', data=df)

# Heatmap — great for correlation matrices
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')

# Pair plot — scatter matrix of all numeric columns
sns.pairplot(df, hue='course')

# Count plot — frequency of categories
sns.countplot(x='course', data=df, palette='Set2')
```

---

## Key Parameters

| Param | Meaning | Example |
|-------|---------|---------|
| `color` | Single color | `'steelblue'` |
| `palette` | Color set | `'Set1'`, `'Blues'`, `'viridis'` |
| `hue` | Color by category | `hue='course'` |
| `figsize` | Chart size | `figsize=(10, 6)` |
| `alpha` | Transparency 0–1 | `alpha=0.7` |
| `annot` | Show values on heatmap | `annot=True` |

---

## Saving Charts

```python
plt.savefig('output.png', dpi=150, bbox_inches='tight')
# dpi=150 → medium resolution
# bbox_inches='tight' → no clipping
```

---

## Matplotlib vs Seaborn Cheat Sheet

| Chart | Matplotlib | Seaborn |
|-------|-----------|---------|
| Bar | `ax.bar()` | `sns.barplot()` |
| Line | `ax.plot()` | `sns.lineplot()` |
| Histogram | `ax.hist()` | `sns.histplot()` |
| Scatter | `ax.scatter()` | `sns.scatterplot()` |
| Heatmap | manual | `sns.heatmap()` |
| Box | `ax.boxplot()` | `sns.boxplot()` |
