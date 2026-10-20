# Week 9 — Model Evaluation

## 🎯 Topics
- **Train / Test Split**: Holdout method, preventing data leakage
- **Cross-Validation**: K-Fold cross-validation, variance reduction
- **Confusion Matrix**: True Positives, False Positives, True Negatives, False Negatives
- **Evaluation Metrics**: Accuracy, Precision, Recall, and $F_1$-Score
- **ROC Curve & AUC**: Receiver Operating Characteristic and Area Under the Curve

## 🗂️ Structure
```
Week9/
├── README.md
├── notes/
│   └── model-evaluation.md             ← Concise guide with formulas and visual diagrams
└── mini-project/
    ├── requirements.txt
    ├── evaluate.py                     ← Clean Python script benchmarking models
    ├── visual_evaluation_metrics.png   ← Saved Confusion Matrix & ROC chart
    └── webpage/                        ← Simple interactive threshold visualizer
        ├── index.html
        ├── style.css
        └── app.js
```

## 🚀 Mini-Project: Model Evaluator
A straightforward Python script that:
- Splits a dataset into training and test sets
- Performs 5-Fold Cross-Validation to assess stability
- Computes Precision, Recall, and $F_1$-Score across classification thresholds
- Plots and saves a 2-panel chart: `visual_evaluation_metrics.png` (Confusion Matrix Heatmap + ROC-AUC curve)

Includes an interactive web dashboard in `webpage/index.html` to explore the precision-recall tradeoff with a live slider.

## 🔧 Setup & Run
```bash
cd Week9/mini-project
pip install -r requirements.txt
python evaluate.py
```

## ⏱️ Duration: 3 Hours
- **Hour 1**: Train/Test splits and K-Fold Cross-Validation
- **Hour 2**: Confusion Matrix, Precision, Recall, and $F_1$-Score
- **Hour 3**: ROC Curves, AUC scores, and hands-on mini-project
