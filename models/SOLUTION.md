# Technical Report — Insurance Bundle Prediction (v7)

> **Score: 0.4875** · Macro F1 = 0.5577 · Latency = 1.18 s · Size = 1.86 MB

---

## Summary

| Aspect | Detail |
|---|---|
| **Model** | LightGBM (multiclass softmax, 75 trees, 31 leaves) |
| **Features** | 62 (44 cleaned + 20 multi-class target-encoded − 2 dropped) |
| **Resampling** | 3-step ROS → SMOTE pipeline for 5 minority classes |
| **CV** | 5-fold StratifiedKFold, OOF Macro F1 = 0.6418 |
| **Explainability** | SHAP TreeExplainer + permutation importance + LightGBM gain |

---

## Where Everything Is

The full technical report, with visualisations and detailed rationale, lives in the notebooks:

| Section | See |
|---|---|
| Problem & scoring | `notebook/technical_report.ipynb` §1 |
| Data cleaning | `notebook/cleaning.ipynb` |
| EDA | `notebook/data_visualization.ipynb` |
| Feature engineering | `notebook/technical_report.ipynb` §3 |
| Class imbalance strategy | `notebook/technical_report.ipynb` §4 |
| Model architecture & HPs | `notebook/technical_report.ipynb` §5–§6 |
| Explainability (SHAP) | `notebook/modelling_best_model_v7.ipynb` §8.5 + `technical_report.ipynb` §7 |
| Results & score decomposition | `notebook/technical_report.ipynb` §8–§9 |
| What didn't work | `notebook/technical_report.ipynb` §10 |
| Submission code | `submissions/submission_v7/solution.py` |

---

## Key Decisions

1. **LightGBM over XGBoost/CatBoost** — 3× faster, native categorical support, smallest footprint.
2. **Multi-class target encoding** — 10 P(class=k|category) columns per categorical, capturing broker specialisation patterns invisible to label encoding.
3. **3-step resampling** — ROS first (C8/C9 from 5→200), then SMOTE (C8/C9→600, C0/C5/C6→3000). Synthetic-only SMOTE fails below ~15 samples.
4. **75 trees, not 150** — Halving trees cut latency from 2.1 s to 1.18 s with only 0.003 F1 drop. Latency penalty matters more.
5. **Dropped per-class thresholds** — v1 (0.3331) used thresholds that overfit; v2 raw argmax (0.4583) was strictly better.

---

## Reproduce

```bash
# Run the best model notebook end-to-end:
jupyter nbconvert --execute notebook/modelling_best_model_v7.ipynb

# Or run the submission directly:
cd submissions/submission_v7
pip install -r requirements.txt
python solution.py
```
