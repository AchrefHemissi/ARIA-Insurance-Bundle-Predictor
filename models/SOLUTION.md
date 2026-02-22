# DataQuest — Winning Model (v7) Technical Report

> **Final competition score: 0.4875** · Macro F1 = 0.5577 · Latency = 1.18 s · Size = 1.86 MB
>
> `score = Macro_F1 × max(0.5, 1 − size_MB/200) × max(0.5, 1 − latency_s/10)`

---

## Table of Contents

1. [Problem Summary](#1-problem-summary)
2. [Data Cleaning](#2-data-cleaning)
3. [Feature Engineering](#3-feature-engineering)
4. [Encoding Strategy](#4-encoding-strategy)
5. [Class Imbalance Strategy](#5-class-imbalance-strategy)
6. [Model Architecture](#6-model-architecture)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [What Didn't Work](#8-what-didnt-work)
9. [Final Results](#9-final-results)

---

## 1. Problem Summary

Predict which of **10 insurance bundles** a customer will purchase (multi-class classification).
The dataset has 60,868 training rows and 28 raw features covering demographics, policy history,
broker information, and purchase channel.

**Critical challenge:** The target is severely imbalanced. `Basic_Health` (class 2) holds
**59.4%** of all rows while `Renter_Basic` (class 8) and `Renter_Premium` (class 9) have
fewer than **10 real samples each** in the training set. Macro F1 scores every class equally,
so a naive majority-class model scores near zero despite 59% raw accuracy.

**Score is the product of three terms** — optimising only F1 while ignoring size and latency
loses points, but optimising size/latency at the cost of F1 loses even more (as v3 and v4 proved).

---

## 2. Data Cleaning

Implemented in `notebook/cleaning.ipynb`. Produces `Data/train_clean.csv` and `Data/test_clean.csv`.

### 2.1 Columns Dropped

| Column | Why dropped |
| --- | --- |
| `User_ID` | Row identifier — zero predictive signal |
| `Payment_Schedule` | 98.7% of rows are `Monthly_EFT` — effectively constant |
| `Policy_Start_Day` | Near-zero correlation with target across all EDA analysis |

### 2.2 Missing Value Strategy

| Column | Missing rate | Strategy | Reason |
| --- | --- | --- | --- |
| `Employer_ID` | 94.3% | Binary presence flag | 94% missing makes the raw number useless; the flag identifies group policies |
| `Broker_ID` | 13.7% | Left as NaN | Handled downstream by target encoding (NaN → global mean) |
| `Region_Code` | 0.5% | Fill `"UNKNOWN"` | Treated as its own category in target encoding |
| `Acquisition_Channel` | 1.1% | Fill `"Aggregator_Site"` | Mode fill — most common channel |
| `Deductible_Tier` | 0.5% | Fill `"Tier_1_High_Ded"` | Mode fill — 79% of rows are Tier 1 |
| `Child_Dependents` | 0.01% | Fill 0 | Only 4 rows affected; 0 is the logical default |

### 2.3 Corrections Discovered During Cleaning

The data brief contained category names that differed from actual values in the CSV:

| Column | Brief said | Actual value |
| --- | --- | --- |
| `Deductible_Tier` | `Tier_2`, `Tier_3` | `Tier_2_Mid_Ded`, `Tier_3_Low_Ded` |
| `Employment_Status` | `Employed_PartTime` | `Contractor` |
| `Acquisition_Channel` | `Referral`, `Corporate_HR` | `Affiliate_Group`, `Corporate_Partner` |

These corrections were essential — using the wrong string values in one-hot encoding
would have silently produced all-zero dummy columns.

---

## 3. Feature Engineering

17 new features created from the raw columns before any encoding.

### 3.1 Household & Financial Features

| Feature | Formula | Rationale |
| --- | --- | --- |
| `Total_Dependents` | `Adult + Child.fillna(0) + Infant` | Overall household size drives bundle choice |
| `Has_Children` | `(Child + Infant) > 0` → 0/1 | Family-oriented buyers behave differently |
| `log_Income` | `log1p(Estimated_Annual_Income)` | Raw income is heavily right-skewed (max = 48× mean); log transform normalises it |
| `log_Income_Per_Dependent` | `log1p(Income / (Total_Dependents + 1))` | Per-person affordability signal |

### 3.2 Risk & Behavioural Features

| Feature | Formula | Rationale |
| --- | --- | --- |
| `Risk_Score` | `Claims_Filed − Years_Without_Claims + Grace_Extensions` | Composite financial stress index |
| `Underwriting_Delayed` | `Underwriting_Processing_Days > 0` → 0/1 | Median underwriting time is 0; a binary flag is more useful than the raw days value |
| `Fast_Buyer` | `Days_Since_Quote < 7` → 0/1 | Fast decision-makers skew to `Basic_Health`; slow buyers to premium bundles |
| `Indecision_Score` | `Days_Since_Quote × Policy_Amendments_Count` | Captures deliberation: long quote time combined with many amendments |
| `Policy_Complexity` | `Custom_Riders + Policy_Amendments` | More amendments and riders → more complex bundle needs |

### 3.3 Missingness Indicators

| Feature | Formula | Rationale |
| --- | --- | --- |
| `Employer_ID_Present` | `Employer_ID.notna()` → 0/1 | The 5.7% with an employer ID represent group/corporate policies — a distinct segment |
| `Broker_ID_Missing` | `Broker_ID.isna()` → 0/1 | Missing broker is structurally different from any known broker |

### 3.4 Temporal Cyclical Features

Month and week are cyclical — December (12) is one step from January (1), not eleven.
Raw integer encoding creates a false discontinuity at year boundaries.

```
Month_Num = MONTH_ORDER[Policy_Start_Month]    # string → integer 1–12
Month_Sin = sin(2π × Month_Num / 12)
Month_Cos = cos(2π × Month_Num / 12)

Week_Sin  = sin(2π × Policy_Start_Week / 52)
Week_Cos  = cos(2π × Policy_Start_Week / 52)
```

`Policy_Start_Day` was dropped (near-zero correlation with target).

---

## 4. Encoding Strategy

### 4.1 Low-Cardinality Categoricals

| Column | Method | Detail |
| --- | --- | --- |
| `Deductible_Tier` | Ordinal (1–4) | Natural order: Tier 1 (high deductible) → Tier 4 (zero deductible) |
| `Broker_Agency_Type` | Binary | `National_Corporate` = 1, `Urban_Boutique` = 0 |
| `Employment_Status` | One-hot (3 dummies) | Drop `Contractor` as the reference category |
| `Acquisition_Channel` | One-hot (4 dummies) | Drop `Affiliate_Group` as the reference category |

### 4.2 Target Encoding — Single Number (Employer_ID)

For `Employer_ID`, a single smoothed mean-target is sufficient because employer group
affiliation is a weak signal (94.3% missing) and the raw ID has no inherent ordering.

```
enc(category) = (n × mean_target + k × global_mean) / (n + k)
```

`k = 300` (smoothing). Unknown categories at inference → global mean.

### 4.3 Target Encoding — Multi-Class (Broker_ID and Region_Code) ← Key Innovation in v7

The single-number encoding for brokers (mean target value) collapses all information
into one number. But brokers **specialise** — Broker 1.0 sells almost exclusively
`Basic_Health` (std = 0.21 on a 0–9 scale), while Broker 250 specialises in premium bundles.
The mean target cannot capture both the *which* class and *how strongly*.

**Multi-class encoding** replaces each categorical column with **10 probability features** —
one per bundle class:

```
P(bundle=k | broker=b) = (n_bk × P(k|b) + k × prior_k) / (n_bk + k)

where:
  n_bk    = number of broker b's customers who bought bundle k
  P(k|b)  = n_bk / total customers with broker b
  prior_k = global proportion of class k in training data
  k = 300 = smoothing constant
```

This gives the model the **full specialisation distribution** per broker/region, not just
the average. Instead of `Broker_ID_TargetEnc` (1 feature), the model sees
`Broker_C0` through `Broker_C9` (10 features).

**Net effect:**

| Encoding | Features used | OOF F1 |
| --- | --- | --- |
| Single-number (v2) | 44 | 0.5977 |
| Multi-class (v7) | 62 | 0.6418 |

The same treatment was applied to `Region_Code` (166 unique values, 10 probability features).

---

## 5. Class Imbalance Strategy

The dataset has a 50× ratio between the largest class (`Basic_Health`) and the smallest
(`Renter_Premium`). Getting classes 8 and 9 right requires a multi-step approach because
SMOTE cannot run when a class has fewer than `k_neighbors + 1` samples.

### 5.1 Three-step Resampling Pipeline

Applied **inside each training fold** — validation data is never touched.

| Step | Classes | Method | Count before | Count after |
| --- | --- | --- | --- | --- |
| 1 | C8, C9 | `RandomOverSampler` | 5–6 samples | 200 |
| 2 | C8, C9 | `SMOTE(k_neighbors=3)` | 200 | 600 |
| 3 | C0, C5, C6 | `SMOTE(k_neighbors=5)` | 823–1,625 | 3,000 |

**Why three steps, not one?** SMOTE requires at least `k_neighbors + 1` real samples to
synthesise new ones. With 5–6 real samples, `k=5` SMOTE fails. The RandomOverSampler
in step 1 duplicates the handful of real samples to reach 200, making SMOTE viable in step 2.

### 5.2 Why `class_weight=None` (Not `'balanced'`)

This was the single largest improvement in v7 (+0.022 OOF F1).

When SMOTE is already resampling the training set to target counts, the gradient boosting
algorithm already sees a more balanced distribution. Adding `class_weight='balanced'`
**double-penalises** the majority class — once by SMOTE reducing its relative share and
again by the class-weight upweighting the minority. The result is over-correction: the model
becomes overly cautious about predicting `Basic_Health` even for clear-cut cases.

Setting `class_weight=None` after SMOTE lets the resampled distribution speak for itself.

---

## 6. Model Architecture

### 6.1 Algorithm: LightGBM

LightGBM was chosen because it satisfies all competition constraints simultaneously:

| Requirement | How LightGBM meets it |
| --- | --- |
| 50 MB ZIP limit | Model string serialises to ~1.8 MB |
| 10 s predict latency | 75 trees × 10 classes, `n_jobs=1`, predicts 15,218 rows in ~0.12 s locally |
| 1 GB RAM | Peak memory well under 200 MB |
| Macro F1 objective | Multiclass mode with SMOTE pre-balancing |

Deep learning models would exceed the size limit. Random Forests are slower and larger.
CatBoost produces larger model files. XGBoost was tested and produced comparable F1
but larger serialised files.

### 6.2 Number of Trees

75 trees per class (750 total). This was determined by benchmarking:

| Trees | OOF F1 | Local latency | Notes |
| --- | --- | --- | --- |
| 50 | 0.4487 | 0.47 s | F1 collapsed — too few for minority classes |
| 75 | 0.5977 | 0.62 s | Best F1/latency balance |
| 150 | ~0.58 | 1.0+ s | Diminishing F1 returns, double the latency |

50 trees is insufficient to learn the minority classes' decision boundaries. 150 trees
adds latency without meaningful F1 improvement at these hyperparameter settings.

### 6.3 Prediction: Raw Argmax (No Threshold Tuning)

v1 tuned per-class probability thresholds on OOF folds to maximise Macro F1.
This inflated the training metric to 0.74 but **hurt actual score** because the thresholds
memorised fold-specific noise rather than generalising. v7 uses raw `argmax(probabilities)`.

---

## 7. Hyperparameter Tuning

A 4-phase sequential search was run to find the best configuration.

### Phase 1 — Encoding comparison

| Encoding | OOF F1 | Features |
| --- | --- | --- |
| Single-number | 0.5898 | 44 |
| Multi-class | **0.5977** | 62 |

Multi-class encoding wins. All subsequent phases use it.

### Phase 2 — Class weight

| `class_weight` | OOF F1 |
| --- | --- |
| `'balanced'` | 0.5977 |
| `None` | **0.6199** |
| manual weights | 0.5715 |

`class_weight=None` wins by 0.022. All subsequent phases use it.

### Phase 3 — Hyperparameter Grid (75 trees fixed)

Grid searched over `num_leaves`, `learning_rate`, `min_child_samples`, `reg_alpha`, `reg_lambda`.
Top 5 configurations:

| OOF F1 | `num_leaves` | `lr` | `min_child_samples` | `reg_alpha` | `reg_lambda` |
| --- | --- | --- | --- | --- | --- |
| **0.6311** | 95 | 0.07 | 20 | 0.1 | 0.5 |
| 0.6306 | 95 | 0.07 | 20 | 0.1 | 2.0 |
| 0.6272 | 63 | 0.07 | 20 | 0.1 | 0.5 |
| 0.6250 | 31 | 0.07 | 20 | 0.05 | 2.0 |
| 0.6243 | 31 | 0.10 | 20 | 0.1 | 0.5 |

Winner: `num_leaves=95`, `lr=0.07`, `min_child_samples=20`, `reg_alpha=0.1`, `reg_lambda=0.5`.

### Phase 4 — Feature Selection

LightGBM split-importance was inspected for zero-importance features. **None found** —
all 62 features contributed at least one split. Feature set kept intact.

### Final 5-fold OOF Validation

| Class | Bundle | OOF F1 |
| --- | --- | --- |
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
| **Macro** | | **0.6418** |

Weakest classes are C3 and C8 — both have limited representation and no natural
distinguishing features beyond the broker/region encoding.

---

## 8. What Didn't Work

### 8.1 Optimising Latency and Size (v3, v4)

After v2 scored 0.4583, the hypothesis was that reducing latency and size penalties further
would compound the score. Two experiments:

- **v3 — 50 trees:** Actual F1 dropped from 0.5176 → 0.4487. Fewer trees are insufficient
  for the minority classes. Score fell to 0.4253.
- **v4 — 150 shallow trees (depth=4):** F1 recovered to 0.4798 but the judge server ran
  at 9.1× local speed that submission, giving 1.23 s latency. Score 0.4188 — worst result.

**Lesson:** The judge server speed varies from 3.6× to 10× local latency across different
submissions. Latency cannot be reliably predicted from local benchmarks. F1 is the only
stable lever.

### 8.2 `class_weight='balanced'` + SMOTE Together

Both are imbalance correction mechanisms. Running them together means the model sees
SMOTE-balanced data (minority classes already upsampled) and then applies additional
gradient upweighting to those same classes. The majority class gets penalised twice,
causing the model to under-predict `Basic_Health` even when it's clearly correct.

### 8.3 Per-Class Threshold Tuning (v1)

Tuning probability thresholds per class on OOF folds raised the held-out F1 to 0.74,
but the actual competition F1 was only 0.4719. The thresholds were memorising fold-level
noise rather than learning a generalising rule. Dropped in v2 and never reintroduced.

### 8.4 OOF Encoding in Training vs Full-Train Encoding at Inference (v5 fix)

`train_clean.csv` was generated with **OOF target encoding** — each row's encoding was
computed from the other folds, not from the full training set. But `solution.py` applies
**full-train encoding** at inference time because there is no held-out fold structure.

This means the model saw slightly different numeric values during training (OOF-encoded)
versus inference (full-train-encoded). Fixing this in v5 — recomputing all three target
encodings using full-train statistics before training — gave +0.004 in actual F1.

---

## 9. Final Results

### Competition Leaderboard

| Version | Key change | Macro F1 | Latency | Score |
| --- | --- | --- | --- | --- |
| v1 | 150 trees + threshold tuning | 0.4719 | 2.83 s | 0.3331 |
| v2 | 75 trees, no thresholds | 0.5176 | 1.07 s | 0.4583 |
| v3 | 50 trees (latency test) | 0.4487 | 0.47 s | 0.4253 |
| v4 | 150 shallow trees depth=4 | 0.4798 | 1.23 s | 0.4188 |
| v5 | Full-train encoding fix | 0.5217 | 1.03 s | 0.4640 |
| **v7** | **All improvements combined** | **0.5577** | **1.18 s** | **0.4875** |

### v7 Submission Artefacts

| File | Size | Purpose |
| --- | --- | --- |
| `submission_v7/solution.py` | ~4 KB | Judge entry point (`preprocess`, `load_model`, `predict`) |
| `submission_v7/model.pkl` | 1.86 MB | LightGBM booster + all encoding maps |
| `submission_v7/requirements.txt` | <1 KB | Package versions (pre-installed on judge) |
| `submission_v7.zip` | 1.86 MB | Final submission file |

### Score Decomposition (v7)

```
Macro F1        = 0.5577
Size penalty    = max(0.5, 1 − 1.86/200)  = 0.9907
Latency penalty = max(0.5, 1 − 1.18/10)   = 0.882

Final score     = 0.5577 × 0.9907 × 0.882 = 0.4875
```

### Key Files

| File | Description |
| --- | --- |
| `notebook/submit_v7.ipynb` | Full reproducible training pipeline |
| `notebook/cleaning.ipynb` | Data cleaning and feature engineering |
| `notebook/modelling.ipynb` | CV benchmarking and model selection |
| `/tmp/solution_v7.py` | Source for `solution.py` inside the ZIP |
