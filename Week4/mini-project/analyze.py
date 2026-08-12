"""
====================================================
Mini-Project: Student Data Analysis
Week 4 — Pandas & Data Visualization
====================================================
Concepts used:
  ✅ Pandas: read CSV, clean, merge, groupby
  ✅ Matplotlib: bar chart, histogram, subplots
  ✅ Seaborn: heatmap, boxplot, distribution plot
====================================================
Run:  python analyze.py
Output: saves 3 PNG chart files in this folder
====================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn theme
sns.set_theme(style="darkgrid", palette="Set2")

# ====================================================
# STEP 1 — Load & Explore the Data
# ====================================================
print("=" * 55)
print("  📂  LOADING DATA")
print("=" * 55)

df = pd.read_csv('students.csv')

print(f"\nShape      : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns    : {list(df.columns)}")
print(f"\nFirst 5 rows:\n")
print(df.head().to_string(index=False))

# ====================================================
# STEP 2 — Clean the Data
# ====================================================
print("\n" + "=" * 55)
print("  🧹  DATA CLEANING")
print("=" * 55)

# Check for missing values
missing = df.isnull().sum()
print(f"\nMissing values:\n{missing[missing > 0] if missing.any() else '  None! Data is clean ✅'}")

# Add computed columns
subjects = ['math', 'science', 'english', 'history', 'computer']

df['average'] = df[subjects].mean(axis=1).round(2)

df['grade'] = df['average'].apply(
    lambda avg: 'A' if avg >= 90 else
                'B' if avg >= 80 else
                'C' if avg >= 70 else
                'D' if avg >= 60 else 'F'
)

df['passed'] = df['average'] >= 60
df['status'] = df['passed'].map({True: 'Pass', False: 'Fail'})

print(f"\nAdded columns: 'average', 'grade', 'passed', 'status'")
print(f"\nUpdated DataFrame (selected cols):\n")
print(df[['name', 'course', 'average', 'grade', 'status']].to_string(index=False))

# ====================================================
# STEP 3 — GroupBy Analysis
# ====================================================
print("\n" + "=" * 55)
print("  📊  GROUP BY COURSE")
print("=" * 55)

course_stats = df.groupby('course').agg(
    students       = ('name',       'count'),
    avg_score      = ('average',    'mean'),
    highest_score  = ('average',    'max'),
    lowest_score   = ('average',    'min'),
    pass_rate_pct  = ('passed',     lambda x: round(x.mean() * 100, 1)),
    avg_attendance = ('attendance', 'mean'),
).round(2)

print(f"\n{course_stats.to_string()}")

# GroupBy year
print("\n" + "─" * 55)
print("  📅  GROUP BY YEAR")
year_stats = df.groupby('year').agg(
    students  = ('name',    'count'),
    avg_score = ('average', 'mean'),
).round(2)
print(f"\n{year_stats.to_string()}")

# ====================================================
# STEP 4 — Visualization 1: Course Performance Bar Chart
# ====================================================
print("\n" + "=" * 55)
print("  📈  GENERATING CHART 1: Course Performance")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Week 4 Mini-Project — Student Analysis", fontsize=14, fontweight='bold')

# Bar chart — avg score by course
colors = ['#4C72B0', '#DD8452', '#55A868']
course_names = course_stats.index.tolist()
avg_scores   = course_stats['avg_score'].tolist()

bars = axes[0].bar(course_names, avg_scores, color=colors, edgecolor='white', linewidth=1.2)
axes[0].set_title("Average Score by Course", fontweight='bold')
axes[0].set_ylabel("Average Score")
axes[0].set_ylim(0, 105)
axes[0].axhline(y=60, color='red', linestyle='--', linewidth=1, label='Pass line (60)')
axes[0].legend()

# Add value labels on bars
for bar, val in zip(bars, avg_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# Bar chart — pass rate by course
pass_rates = course_stats['pass_rate_pct'].tolist()
bars2 = axes[1].bar(course_names, pass_rates, color=colors, edgecolor='white', linewidth=1.2)
axes[1].set_title("Pass Rate by Course (%)", fontweight='bold')
axes[1].set_ylabel("Pass Rate (%)")
axes[1].set_ylim(0, 115)

for bar, val in zip(bars2, pass_rates):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('chart1_course_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: chart1_course_performance.png")

# ====================================================
# STEP 5 — Visualization 2: Grade Distribution + Histogram
# ====================================================
print("\n  📈  GENERATING CHART 2: Grade Distribution")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Grade Distribution Analysis", fontsize=14, fontweight='bold')

# Seaborn histogram of average scores
sns.histplot(df['average'], bins=8, kde=True, ax=axes[0],
             color='steelblue', edgecolor='white')
axes[0].set_title("Distribution of Student Averages", fontweight='bold')
axes[0].set_xlabel("Average Score")
axes[0].set_ylabel("Count")
axes[0].axvline(x=df['average'].mean(), color='red', linestyle='--',
                label=f"Mean: {df['average'].mean():.1f}")
axes[0].legend()

# Grade count bar chart
grade_order = ['A', 'B', 'C', 'D', 'F']
grade_counts = df['grade'].value_counts().reindex(grade_order, fill_value=0)
grade_colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']

axes[1].bar(grade_counts.index, grade_counts.values,
            color=grade_colors, edgecolor='white', linewidth=1.2)
axes[1].set_title("Number of Students per Grade", fontweight='bold')
axes[1].set_xlabel("Grade")
axes[1].set_ylabel("Number of Students")

for i, (grade, count) in enumerate(zip(grade_counts.index, grade_counts.values)):
    axes[1].text(i, count + 0.1, str(count), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('chart2_grade_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: chart2_grade_distribution.png")

# ====================================================
# STEP 6 — Visualization 3: Seaborn Heatmap (Correlation)
# ====================================================
print("\n  📈  GENERATING CHART 3: Subject Correlation Heatmap")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Subject Analysis", fontsize=14, fontweight='bold')

# Correlation heatmap
corr_cols = subjects + ['attendance', 'average']
corr = df[corr_cols].corr().round(2)

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
            ax=axes[0], linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.8})
axes[0].set_title("Subject Correlation Heatmap", fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

# Seaborn boxplot — score by course
df_melted = df.melt(id_vars=['course'], value_vars=subjects,
                    var_name='subject', value_name='score')
sns.boxplot(x='subject', y='score', hue='course', data=df_melted,
            ax=axes[1], palette='Set2')
axes[1].set_title("Score Distribution by Subject & Course", fontweight='bold')
axes[1].set_xlabel("Subject")
axes[1].set_ylabel("Score")
axes[1].legend(title='Course', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('chart3_subject_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: chart3_subject_heatmap.png")

# ====================================================
# STEP 7 — Final Summary
# ====================================================
print("\n" + "=" * 55)
print("  🏆  FINAL SUMMARY")
print("=" * 55)

top3 = df.nlargest(3, 'average')[['name', 'course', 'average', 'grade']]
bottom3 = df.nsmallest(3, 'average')[['name', 'course', 'average', 'grade']]

print(f"\n  Total Students : {len(df)}")
print(f"  Class Average  : {df['average'].mean():.2f}")
print(f"  Highest Score  : {df['average'].max():.2f} ({df.loc[df['average'].idxmax(), 'name']})")
print(f"  Lowest Score   : {df['average'].min():.2f} ({df.loc[df['average'].idxmin(), 'name']})")
print(f"  Pass Rate      : {df['passed'].mean()*100:.1f}%")

print(f"\n  🥇 Top 3 Students:")
print(top3.to_string(index=False))

print(f"\n  ⚠️  Bottom 3 Students:")
print(bottom3.to_string(index=False))

print("\n" + "=" * 55)
print("  ✅  Analysis complete! Check the PNG files.")
print("=" * 55 + "\n")
