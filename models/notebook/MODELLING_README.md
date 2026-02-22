# Notebooks Guide

> Navigate the modelling pipeline across notebooks.

---

## Notebook Order

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `cleaning.ipynb` | Raw → cleaned CSVs (missing values, outliers, 17 engineered features) |
| 2 | `data_visualization.ipynb` | EDA — distributions, correlations, class imbalance, broker patterns |
| 3 | `modelling.ipynb` | Initial exploration: v1 (thresholds) and v2 (argmax baseline) |
| 4 | `modelling_best_model_v7.ipynb` | **Best model** — training, CV, SHAP explainability, model export |
| 5 | `technical_report.ipynb` | **Full technical report** — rationale, visualisations, rubric coverage |

---

## Pipeline at a Glance

```
raw CSVs
  → cleaning.ipynb          (impute, encode, engineer 17 features)
  → train_clean.csv
  → modelling_best_model_v7 (add 20 target-encoded features, SMOTE/ROS, LightGBM, SHAP)
  → model.pkl + submission.csv
```

---

## Rubric Mapping

| Rubric Area (Weight) | Primary Notebook |
|---|---|
| Feature Engineering (20) | `cleaning.ipynb` + `technical_report.ipynb` §3 |
| Model Performance (30) | `modelling_best_model_v7.ipynb` + `technical_report.ipynb` §5–§8 |
| Explainability (10) | `modelling_best_model_v7.ipynb` §8.5 |
| Technical Report (10) | `technical_report.ipynb` |
