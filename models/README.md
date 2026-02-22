# DataQuest — Insurance Bundle Prediction

> **Final Score: 0.4875** · Macro F1 = 0.5577 · LightGBM · 62 features · SHAP explainability

---

## Quick Start

```bash
cd submissions/submission_v7
pip install -r requirements.txt
python solution.py        # writes submission.csv
```

---

## Project Structure

```
models/
├── README.md                  ← You are here
├── SOLUTION.md                ← One-page technical summary
├── Data/
│   ├── raw/                   ← train.csv, test.csv (competition download)
│   └── processed/             ← train_clean.csv, test_clean.csv
├── notebook/
│   ├── technical_report.ipynb ← Full technical report with visualisations
│   ├── cleaning.ipynb         ← Data cleaning pipeline
│   ├── data_visualization.ipynb ← Exploratory data analysis
│   ├── modelling.ipynb        ← Initial model exploration (v1–v2)
│   └── modelling_best_model_v7.ipynb ← Best model training + explainability
├── submissions/
│   └── submission_v7/         ← Final submission (solution.py)
└── test/
    └── test_latency.py        ← Latency benchmark
```

---

## Notebooks — Where to Look

| Rubric Area (Weight) | Notebook |
|---|---|
| **Technical Report (10)** | `notebook/technical_report.ipynb` |
| **Feature Engineering (20)** | `notebook/cleaning.ipynb` → `technical_report.ipynb` §3 |
| **Model Performance (30)** | `notebook/modelling_best_model_v7.ipynb` → `technical_report.ipynb` §5–§8 |
| **Explainability (10)** | `notebook/modelling_best_model_v7.ipynb` §8.5 → `technical_report.ipynb` §7 |
| **EDA** | `notebook/data_visualization.ipynb` |

---

## Scoring Formula

```
final_score = Macro_F1 × max(0.5, 1 − size_MB/200) × max(0.5, 1 − latency_s/10)
```
