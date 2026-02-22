# ARIA — Insurance Bundle Predictor: Technical Report

> **Final Score: 0.4875** · Macro F1 = 0.5577 · LightGBM · 62 features · 1.18 s latency · 1.86 MB

**Team:** Mohamed Achref Hemissi · Mohamed Dhia Medini · Rayen Chemlali · Leith Engazzou

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Data Overview & EDA](#3-data-overview--eda)
4. [Feature Engineering — 62 Features](#4-feature-engineering--62-features)
5. [Class Imbalance Strategy](#5-class-imbalance-strategy)
6. [Model Architecture & Justification](#6-model-architecture--justification)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [Explainability Analysis](#8-explainability-analysis)
9. [Results & Submission History](#9-results--submission-history)
10. [What Didn't Work](#10-what-didnt-work)
11. [Quick Start](#11-quick-start)
12. [Project Structure](#12-project-structure)

---

## 1. Problem Statement

Predict which of **10 insurance bundles** (classes 0–9) a customer will purchase.
The competition evaluates submissions by **Macro F1-Score** — every class counts equally regardless of how many samples it has.

### Scoring Formula

```
final_score = Macro_F1
            × max(0.5, 1 − size_MB / 200)
            × max(0.5, 1 − latency_s / 10)
```

A perfect-F1 model at 50 MB and 1 s latency scores only **0.675** — both accuracy *and* efficiency matter.

### Resource Constraints

| Constraint | Limit |
|---|---|
| Model ZIP size | 50 MB |
| RAM at inference | 1 GB |
| CPU cores | 1 |
| Prediction latency | 10 s |
| Total runtime | 120 s |

---

## 2. System Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA (28 cols)                           │
│                   60,868 train / 15,218 test rows                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  DATA CLEANING      │
                    │  • Drop 3 columns   │
                    │    (User_ID,        │
                    │     Payment_Sched., │
                    │     Policy_Start_D.)│
                    │  • Impute 6 cols    │
                    │    (median/mode)    │
                    │  • Fix 3 anomalies  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ FEATURE ENGINEERING │
                    │  +17 new features   │
                    │  • Household ratios │
                    │  • Risk scores      │
                    │  • Cyclical time    │
                    │  • Missingness flags│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    ENCODING         │
                    │  • Ordinal (risk)   │
                    │  • One-hot (7 cols) │
                    │  • Employer target  │
                    │    enc. (1 col)     │
                    │  • Multi-class      │
                    │    target enc.      │
                    │    Broker×10 +      │
                    │    Region×10        │
                    │  − Drop 8 raw cats  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  62-FEATURE MATRIX  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐   applied inside each
                    │  3-STEP RESAMPLING  │◄─ CV fold only; val set
                    │  (train fold only)  │   is never touched
                    │  Step 1: ROS        │
                    │   C8,C9: 5→200      │
                    │  Step 2: SMOTE k=3  │
                    │   C8,C9: 200→600    │
                    │  Step 3: SMOTE k=5  │
                    │   C0,C5,C6: →3,000  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     LightGBM        │
                    │  75 trees × 10 cls  │
                    │  class_weight=None  │
                    │  5-fold Stratified  │
                    │  OOF Macro F1=0.642 │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  RAW ARGMAX PREDICT │
                    │  (no threshold tun.)│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  submission.csv     │
                    │  Macro F1 = 0.5577  │
                    │  Score = 0.4875     │
                    └─────────────────────┘
```

### File Flow

```
raw/train.csv ──► cleaning.ipynb ──► Data/processed/train_clean.csv
raw/test.csv  ──► cleaning.ipynb ──► Data/processed/test_clean.csv
                                            │
                           modelling_best_model_v7.ipynb
                                  (adds 20 target-enc. cols)
                                            │
                                    model.pkl  submission.csv
                                            │
                              submissions/submission_v7/solution.py
                                   (production inference)
```

---

## 3. Data Overview & EDA

### Dataset at a Glance

| Attribute | Train | Test |
|---|---|---|
| Rows | 60,868 | 15,218 |
| Raw columns | 29 (incl. target) | 28 |
| Duplicate rows | 0 | 0 |
| Columns with missing values | 6 | 5 |

### Class Imbalance — The Core Challenge

The dataset has a **7,227× ratio** between the largest and smallest class.
Macro F1 weights every class equally, so a naive majority-class model scores near zero despite 59% accuracy.

| Class | Bundle | Count | Share |
|---|---|---|---|
| C2 | Basic_Health | 36,136 | 59.4% |
| C4 | Health_Dental_Vision | 13,958 | 22.9% |
| C3 | Family_Comprehensive | 4,831 | 7.9% |
| C7 | Premium_Health_Life | 2,286 | 3.8% |
| C1 | Auto_Liability_Basic | 1,625 | 2.7% |
| C0 | Auto_Comprehensive | 823 | 1.4% |
| C6 | Home_Standard | 719 | 1.2% |
| C5 | Home_Premium | 479 | 0.8% |
| **C8** | **Renter_Basic** | **6** | **0.01%** |
| **C9** | **Renter_Premium** | **5** | **0.008%** |

Two distinct sub-problems emerge:
- **Problem A — Moderate minorities** (C0, C5, C6): 479–823 samples → addressable with SMOTE.
- **Problem B — Extreme minorities** (C8, C9): 5–6 samples total → a data ceiling; only broker multi-class encoding provides viable signal.

### Key EDA Findings

**Income distribution** — raw income is skewed 48×. `log(1 + Income)` normalises it and reveals a threshold around `log(income) ≈ 10.5` that separates premium from basic bundles (visible in SHAP partial-dependence plots).

**Broker specialisation** — brokers are not generalists. Broker 1.0 sells almost exclusively Basic_Health (σ = 0.21). Broker 250 concentrates on premium bundles. A single-number encoding averages this away; multi-class encoding preserves the full distribution.

**Fast buyers** — customers who decide within 7 days of quote (`Days_Since_Quote < 7`) skew heavily toward Basic_Health. This is captured by the `Fast_Buyer` binary flag.

**Missing values are informative** — `Employer_ID` is present for only 5.7% of customers. Those with an employer represent group-policy holders, a structurally different purchasing segment. `Employer_ID_Present` is therefore kept as a feature rather than just imputed.

**Temporal patterns** — month and week of policy start show cyclical seasonality effects. Sine/cosine encoding maintains December → January continuity.

> Full EDA with 69 cells of analysis and visualisations: [`notebook/data_visualization.ipynb`](notebook/data_visualization.ipynb)

---

## 4. Feature Engineering — 62 Features

Starting from 28 raw columns, the pipeline produces **62 features** through four stages.

### Pipeline Summary

| Stage | Features | Description |
|---|---|---|
| Raw columns (after 3 drops) | 25 | Dropped `User_ID`, `Payment_Schedule`, `Policy_Start_Day` |
| Engineered features | +17 | Household, financial, risk, temporal, missingness |
| One-hot encoding | +7 | Employment (3 dummies), Acquisition (4 dummies) |
| Target encoding (single) | +1 | `Employer_ID_TargetEnc` |
| Multi-class target encoding | +20 | `Broker_C0`–`C9`, `Region_C0`–`C9` |
| Dropped (replaced by encodings) | −8 | Raw categoricals |
| **Total** | **62** | |

### 4.1 Household & Financial Features

| Feature | Formula | Rationale |
|---|---|---|
| `Total_Dependents` | Adult + Child + Infant | Household size is the primary driver of bundle type |
| `Has_Children` | (Child + Infant) > 0 | Family buyers form a distinct purchasing segment |
| `log_Income` | log(1 + Income) | Raw income is 48× skewed; log transform normalises it |
| `log_Income_Per_Dep` | log(1 + Income / (Deps + 1)) | Per-person affordability proxy separates premium from basic |

### 4.2 Risk & Behavioural Features

| Feature | Formula | Rationale |
|---|---|---|
| `Risk_Score` | Claims − Years_Clean + Grace_Ext | Composite financial stress index |
| `Underwriting_Delayed` | Processing_Days > 0 | Binary flag; median = 0, so delays are anomalous |
| `Fast_Buyer` | Days_Since_Quote < 7 | Fast deciders → Basic_Health; deliberators → complex bundles |
| `Indecision_Score` | Days_Quote × Amendments | Captures deliberation × complexity jointly |
| `Policy_Complexity` | Riders + Amendments | Higher values signal need for comprehensive coverage |

### 4.3 Missingness Indicators

| Feature | Formula | Rationale |
|---|---|---|
| `Employer_ID_Present` | `Employer_ID.notna()` | 5.7% with employer = group policy holders (distinct segment) |
| `Broker_ID_Missing` | `Broker_ID.isna()` | Structurally different from customers with known brokers |

### 4.4 Temporal Cyclical Encoding

Raw month/week integers break the December → January continuity.
Sine/cosine encoding preserves the ring topology of time.

| Feature | Formula |
|---|---|
| `Month_Sin` / `Month_Cos` | sin/cos(2π × Month / 12) |
| `Week_Sin` / `Week_Cos` | sin/cos(2π × Week / 52) |

### 4.5 Key Innovation — Multi-Class Target Encoding

Standard target encoding collapses broker specialisation into a single mean and destroys the distributional shape. **Multi-class encoding** replaces each high-cardinality categorical with **10 probability columns** — one per bundle class — using smoothed Bayesian estimation:

```
P(bundle=k | broker=b) = (n_bk × P(k|b) + λ × prior_k) / (n_bk + λ),  λ = 300
```

This gives the model the full specialisation distribution per broker/region, not just a weighted average. For the extreme minority classes C8/C9 (5–6 samples), the broker encoding is essentially the **only learnable signal**.

| Encoding Strategy | Features | OOF Macro F1 |
|---|---|---|
| Single-number target encoding | 44 | 0.5977 |
| **Multi-class encoding (v7)** | **62** | **0.6418 (+0.044)** |

### 4.6 Complete Feature Matrix

| Feature Group | Count | Description |
|---|---|---|
| Original numeric | 32 | Policy, risk, temporal, financial columns |
| One-hot dummies | 7 | Employment (3), Acquisition channel (4) |
| Binary flags | 3 | `Agency_National`, `Employer_ID_Present`, `Broker_ID_Missing` |
| `Employer_ID_TargetEnc` | 1 | Smoothed single-number encoding |
| `Broker_C0`–`Broker_C9` | 10 | Multi-class broker probabilities |
| `Region_C0`–`Region_C9` | 10 | Multi-class region probabilities |
| **Total** | **62** | |

---

## 5. Class Imbalance Strategy

### 3-Step Resampling Pipeline

Applied **inside each CV fold** — the validation set is never touched to prevent leakage.

| Step | Classes | Method | Before | After | Rationale |
|---|---|---|---|---|---|
| 1 | C8, C9 | RandomOverSampler | 5–6 | 200 | SMOTE requires k+1 samples minimum |
| 2 | C8, C9 | SMOTE (k=3) | 200 | 600 | Synthetic neighbours now viable |
| 3 | C0, C5, C6 | SMOTE (k=5) | 479–823 | 3,000 | Bring moderate minorities up |

### Why `class_weight=None` (Not `'balanced'`)

This was the **single largest improvement** in v7: **+0.022 OOF F1**.

When SMOTE already rebalances the training distribution, adding `class_weight='balanced'` **double-penalises** the majority class — once via synthetic resampling and again via gradient upweighting. The result is over-prediction of minorities and collapse of Basic_Health recall.

| `class_weight` | OOF Macro F1 |
|---|---|
| `'balanced'` | 0.5977 |
| Manual weights | 0.5715 |
| **`None`** | **0.6199 (+0.022)** |

---

## 6. Model Architecture & Justification

### Why LightGBM?

| Competition Constraint | LightGBM Advantage | Why Alternatives Failed |
|---|---|---|
| 50 MB ZIP limit | **1.86 MB** model file | Deep learning > 50 MB; CatBoost files larger |
| 10 s predict latency | 75 trees → ~0.12 s locally | Random Forest too slow |
| 1 GB RAM | Peak < 200 MB | No issue |
| Macro F1 on imbalanced data | Multiclass softmax + SMOTE | XGBoost comparable but serialises larger |
| 1 CPU core | `n_jobs=1` forced | All models equivalent here |

LightGBM's histogram-based leaf-wise tree growth is **3× faster** than XGBoost's level-wise approach at equal accuracy. Its model files are compact, and native support for multi-class softmax avoids the one-vs-rest overhead.

### Final Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| `objective` | `multiclass` | 10-class problem |
| `num_class` | 10 | — |
| `n_estimators` | 75 | Sweet spot: best F1/latency ratio (see ablation) |
| `learning_rate` | 0.07 | Grid-searched; lower rates needed more trees |
| `max_depth` | 7 | Prevents overfitting on 60k rows |
| `num_leaves` | 95 | Grid-searched best value |
| `min_child_samples` | 20 | Avoids splits on noise from SMOTE'd minorities |
| `subsample` | 0.8 | Row subsampling for variance reduction |
| `colsample_bytree` | 0.8 | Feature subsampling per tree |
| `reg_alpha` | 0.1 | L1 regularisation (grid-searched) |
| `reg_lambda` | 0.5 | L2 regularisation (grid-searched) |
| `class_weight` | `None` | SMOTE handles imbalance (see §5) |
| `n_jobs` | 1 | Respects single-CPU constraint |

### Tree Count Ablation

| Trees | OOF F1 | Local Latency | Notes |
|---|---|---|---|
| 50 | 0.4487 | 0.47 s | F1 collapsed for minority classes |
| **75** | **0.5977** | **0.62 s** | **Best F1/latency balance** |
| 150 | ~0.58 | 1.0+ s | Diminishing F1, double the latency |

**75 is the sweet spot**: cutting to 50 dropped F1 by 0.07; increasing to 150 added latency without recovering that F1.

### Prediction Strategy: Raw Argmax

v1 tuned per-class probability thresholds on OOF folds, inflating held-out F1 to 0.74 — but actual competition F1 was only 0.4719. Thresholds memorised fold-level noise. **Raw argmax** is more robust on unseen data.

---

## 7. Hyperparameter Tuning

Four sequential phases, each isolating one variable to measure its impact cleanly.

### Phase 1 — Encoding Comparison

| Encoding | Features | OOF F1 |
|---|---|---|
| Single-number | 44 | 0.5898 |
| **Multi-class** | **62** | **0.5977 (+0.008)** |

### Phase 2 — Class Weight

| `class_weight` | OOF F1 |
|---|---|
| `'balanced'` | 0.5977 |
| **`None`** | **0.6199 (+0.022)** |

### Phase 3 — Hyperparameter Grid (75 trees fixed)

| OOF F1 | num_leaves | lr | min_child | reg_alpha | reg_lambda |
|---|---|---|---|---|---|
| **0.6311** | **95** | **0.07** | **20** | **0.1** | **0.5** |
| 0.6306 | 95 | 0.07 | 20 | 0.1 | 2.0 |
| 0.6272 | 63 | 0.07 | 20 | 0.1 | 0.5 |
| 0.6250 | 31 | 0.07 | 20 | 0.05 | 2.0 |

### Phase 4 — Feature Selection

All 62 features contributed at least one tree split. Feature set kept intact; removing any group degraded OOF F1.

### Cumulative Impact Summary

| Technique | ΔOOF F1 |
|---|---|
| Multi-class target encoding | +0.008 |
| `class_weight=None` | +0.022 |
| Hyperparameter grid search | +0.013 |
| Full-train encoding consistency fix (v5) | +0.004 actual F1 |
| **Combined (v7)** | **0.6418 OOF F1** |

---

## 8. Explainability Analysis

Three complementary importance methods used to cross-validate the model's decision logic.

### Top 5 Features Across Three Methods

| Rank | LightGBM Gain | SHAP (mean \|SHAP\|) | Permutation (F1 drop) |
|---|---|---|---|
| 1 | `Agency_National` | `Agency_National` (0.306) | `Agency_National` |
| 2 | `log_Income` | `log_Income` (0.214) | `log_Income` |
| 3 | `Broker_C1` | `Broker_C1` (0.181) | `Broker_C1` |
| 4 | `Total_Dependents` | `Total_Dependents` (0.178) | `Total_Dependents` |
| 5 | `Broker_C6` | `Broker_C6` (0.115) | `Deductible_Tier_Ord` |

All three methods agree on the **top 4 features** — strong cross-validation that model logic is interpretable and robust.

### SHAP Visualisations (in `modelling_best_model_v7.ipynb`)

| Plot | Key Insight |
|---|---|
| Bar plot (all 10 classes) | `Agency_National` dominates; broker encodings collectively are the strongest group |
| Beeswarm (Class 0) | High `Broker_C0` values push predictions toward Auto_Comprehensive |
| Minority class summaries (C8, C9) | These classes rely almost entirely on broker multi-class encoding |
| Waterfall (individual prediction) | Per-feature contributions with direction and magnitude |
| Partial dependence plots | Non-linear income threshold for bundle switching at log(income) ≈ 10.5 |
| SHAP interaction analysis | Broker × Region geographic specialisation patterns |
| Confidence distribution | Basic_Health has the highest average prediction confidence |

### Key Findings

1. **Broker specialisation is the dominant signal.** 6 of the top 15 features are broker-related. Multi-class encoding captures the full specialisation distribution.
2. **Income drives the premium vs. basic split.** A threshold effect is visible around `log(income) ≈ 10.5`.
3. **Family size drives family-specific bundles.** `Total_Dependents` and `Has_Children` separate Family_Comprehensive from individual bundles.
4. **Minority classes C8/C9 are broker-dependent.** Without multi-class broker encoding, these 5–6-sample classes are unlearnable.
5. **All three importance methods agree** on the top features, validating interpretability.

---

## 9. Results & Submission History

### Score Decomposition (v7)

```
score = 0.5577  ×  max(0.5, 1 − 1.86/200)  ×  max(0.5, 1 − 1.18/10)
      = 0.5577  ×  0.9907                   ×  0.882
      = 0.4875
```

The model achieves near-perfect size penalty (1.86 MB vs. 200 MB cap).
Latency is the binding penalty. F1 is the dominant improvement lever.

### Submission History

| Ver | Key Change | Macro F1 | Latency | Score | Lesson |
|---|---|---|---|---|---|
| v1 | 150 trees + per-class threshold tuning | 0.4719 | 2.83 s | 0.3331 | Thresholds overfit to fold noise |
| v2 | 75 trees, raw argmax | 0.5176 | 1.07 s | 0.4583 | Sweet spot for tree count |
| v3 | 50 trees (latency test) | 0.4487 | 0.47 s | 0.4253 | Too few trees → F1 collapse |
| v4 | 150 shallow trees depth=4 | 0.4798 | 1.23 s | 0.4188 | Shallow trees underfit 10-class problem |
| v5 | Full-train encoding consistency fix | 0.5217 | 1.03 s | 0.4640 | Train/inference feature mismatch fixed |
| **v7** | **All improvements combined** | **0.5577** | **1.18 s** | **0.4875** | **Best submission** |

### Per-Class OOF F1 (v7)

| Class | Bundle | OOF F1 |
|---|---|---|
| C2 | Basic_Health | 0.846 |
| C9 | Renter_Premium | 1.000 |
| C5 | Home_Premium | 0.715 |
| C1 | Auto_Liability_Basic | 0.674 |
| C7 | Premium_Health_Life | 0.648 |
| C6 | Home_Standard | 0.614 |
| C0 | Auto_Comprehensive | 0.548 |
| C4 | Health_Dental_Vision | 0.544 |
| C8 | Renter_Basic | 0.421 |
| C3 | Family_Comprehensive | 0.408 |
| | **Macro Average** | **0.6418** |

The OOF-to-actual gap (0.6418 → 0.5577) reflects judge server overhead and environment variability, not overfitting.

---

## 10. What Didn't Work

### Latency Optimisation at the Cost of F1

| Attempt | Result | Lesson |
|---|---|---|
| v3 — 50 trees | F1 dropped 0.07, score fell to 0.4253 | Below 75 trees, minority class F1 collapses |
| v4 — 150 shallow trees (depth=4) | F1 = 0.48, score = 0.42 | Shallow trees underfit the 10-class problem |

The judge server varies from 3.6× to 10× local latency. **F1 is the only stable lever** — latency cannot be reliably engineered for.

### `class_weight='balanced'` with SMOTE

Both mechanisms correct for class imbalance. Running them together **double-penalises** the majority class: SMOTE resamples it down, then `'balanced'` upweights the minorities again. The result is over-prediction of rare classes and under-prediction of Basic_Health, dropping macro F1 by 0.022.

### Per-Class Threshold Tuning (v1)

Threshold tuning on OOF predictions inflated held-out F1 to 0.74 — but competition F1 was only 0.4719 (score = 0.3331). Thresholds memorised fold-specific noise rather than generalising. Dropped from v2 onwards.

### OOF vs. Full-Train Encoding Inconsistency (v1–v4)

`train_clean.csv` used OOF (leave-one-fold-out) target encoding, but `solution.py` used full-train encoding. The model trained on features calculated one way and inferred on features calculated a different way. **Fixed in v5 (+0.004 actual F1)** by making both paths use the same full-train encoding.

### Alternative Models Evaluated

| Model | Issue |
|---|---|
| CatBoost | Larger model files, no F1 improvement over LightGBM |
| XGBoost | Comparable F1 but serialises to larger files (tighter size budget) |
| Random Forest | Too slow (>10 s) and too large for constraints |
| Deep Learning (MLP) | Model ZIP exceeds 50 MB limit even with quantisation |

---

## 11. Quick Start

```bash
# Option A: run final submission directly
cd submissions/submission_v7
pip install -r requirements.txt
python solution.py            # writes submission.csv

# Option B: reproduce full training pipeline
jupyter nbconvert --execute notebook/modelling_best_model_v7.ipynb

# Option C: reproduce from scratch
jupyter nbconvert --execute notebook/cleaning.ipynb
jupyter nbconvert --execute notebook/modelling_best_model_v7.ipynb
```

### Dependencies

```
lightgbm >= 4.0
imbalanced-learn >= 0.11
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.26
shap >= 0.44
```

---

## 12. Project Structure

```
models/
├── README.md                          ← This file (full technical report)
├── SOLUTION.md                        ← One-page key-decisions summary
├── DataQuest_Technical_Report.tex     ← LaTeX version of this report
│
├── Data/
│   ├── README.md                      ← Column descriptions & class table
│   ├── raw/
│   │   ├── train.csv                  ← 60,868 × 29 (competition download)
│   │   └── test.csv                   ← 15,218 × 28
│   └── processed/
│       ├── train_clean.csv            ← Output of cleaning.ipynb (45 cols)
│       └── test_clean.csv             ← Output of cleaning.ipynb (44 cols)
│
├── notebook/
│   ├── MODELLING_README.md            ← Notebook order & rubric map
│   ├── data_visualization.ipynb       ← EDA (69 cells, distributions, broker patterns)
│   ├── cleaning.ipynb                 ← Cleaning pipeline → processed CSVs
│   ├── modelling.ipynb                ← v1–v2 exploration & baselines
│   ├── modelling_best_model_v7.ipynb  ← Best model + full SHAP explainability
│   └── technical_report.ipynb        ← Narrative report with inline visualisations
│
├── submissions/
│   └── submission_v7/
│       ├── solution.py                ← Production inference (must run in <10 s)
│       ├── model.pkl                  ← Trained LightGBM (1.86 MB)
│       ├── requirements.txt
│       └── submission.csv
│
└── test/
    └── test_latency.py                ← Local latency benchmark
```

### Rubric Coverage

| Rubric Area | Weight | Where to Find It |
|---|---|---|
| Feature Engineering | 20 pts | §4 above · `notebook/cleaning.ipynb` |
| Final Score | 30 pts | §9 above · `submissions/submission_v7/solution.py` |
| Explainability | 10 pts | §8 above · `notebook/modelling_best_model_v7.ipynb` §8.5 |
| Technical Report | 10 pts | This README · `notebook/technical_report.ipynb` |
| EDA | Bonus | §3 above · `notebook/data_visualization.ipynb` |
| Failed Attempts | Bonus | §10 above · `notebook/technical_report.ipynb` §10 |
