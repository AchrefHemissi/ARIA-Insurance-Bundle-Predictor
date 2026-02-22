# DataQuest Hackathon — Intelligent Insurance Bundle Recommendation

> **Exploratory Data Analysis Report**
> Multi-class classification · 10 insurance bundles · Evaluated on **Macro F1-Score**

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Overview](#2-dataset-overview)
3. [Column Dictionary](#3-column-dictionary)
4. [Target Variable Analysis](#4-target-variable-analysis)
5. [Missing Values](#5-missing-values)
6. [Demographics & Financials](#6-demographics--financials)
7. [Customer History & Risk Profile](#7-customer-history--risk-profile)
8. [Policy Details & Preferences](#8-policy-details--preferences)
9. [Sales, Brokers & Acquisition](#9-sales-brokers--acquisition)
10. [Temporal Patterns](#10-temporal-patterns)
11. [Correlation Analysis — All Features](#11-correlation-analysis--all-features)
12. [Test Set Analysis](#12-test-set-analysis)
13. [Key Findings Summary](#13-key-findings-summary)
14. [Feature Engineering Recommendations](#14-feature-engineering-recommendations)
15. [Columns to Drop](#15-columns-to-drop)
16. [Model Strategy](#16-model-strategy)

---

## 1. Problem Statement

Build a model that predicts which of **10 insurance bundles** a prospective customer
will purchase, based on their demographic profile, policy history, broker information,
and acquisition metadata.

**Scoring formula:**

```text
final_score = Macro_F1 × max(0.5, 1 − size_MB/200) × max(0.5, 1 − latency_s/10)
```

A perfect F1 model at 50 MB and 1 s latency scores only **0.675** — keep models small and fast.

**Resource limits:** 50 MB zip · 1 GB RAM · 1 CPU core · 120 s total runtime · 10 s predict latency.

---

## 2. Dataset Overview

| Attribute                   | Train   | Test   |
| --------------------------- | ------- | ------ |
| Rows                        | 60,868  | 15,218 |
| Feature columns             | 28      | 28     |
| Target column               | int 0–9 | —      |
| Duplicate rows              | 0       | 0      |
| Columns with missing values | 6       | 5      |

### Column Groups

- **Identifier:** `User_ID`
- **Target:** `Purchased_Coverage_Bundle`
- **Demographics:** `Adult_Dependents`, `Child_Dependents`, `Infant_Dependents`, `Estimated_Annual_Income`, `Employment_Status`, `Region_Code`
- **Risk History:** `Existing_Policyholder`, `Previous_Claims_Filed`, `Years_Without_Claims`, `Previous_Policy_Duration_Months`, `Policy_Cancelled_Post_Purchase`
- **Policy Preferences:** `Deductible_Tier`, `Payment_Schedule`, `Vehicles_on_Policy`, `Custom_Riders_Requested`, `Grace_Period_Extensions`
- **Sales & Underwriting:** `Days_Since_Quote`, `Underwriting_Processing_Days`, `Policy_Amendments_Count`, `Acquisition_Channel`, `Broker_Agency_Type`, `Broker_ID`, `Employer_ID`
- **Temporal:** `Policy_Start_Year`, `Policy_Start_Month`, `Policy_Start_Week`, `Policy_Start_Day`

### Data Types

| Type            | Count |
| --------------- | ----- |
| Integer (int64) | 17    |
| Float (float64) | 4     |
| String (object) | 9     |

- **Float columns:** `Child_Dependents`, `Broker_ID`, `Employer_ID`, `Estimated_Annual_Income`
- **String columns:** `User_ID`, `Region_Code`, `Broker_Agency_Type`, `Deductible_Tier`, `Acquisition_Channel`, `Payment_Schedule`, `Employment_Status`, `Policy_Start_Month`, `Bundle_Name`

> **Note:** `Policy_Start_Month` is stored as a month name string (e.g. `"April"`, `"July"`), not an integer. Must be mapped to a number before modelling.

---

## 3. Column Dictionary

Full description of every column in the dataset.

### Identifier & Target

- **`User_ID`** _(string)_ — Unique customer identifier (e.g. `USR_000000`). No predictive value — drop before training.
- **`Purchased_Coverage_Bundle`** _(int 0–9)_ — **Target.** The insurance bundle the customer purchased. Maps to 10 bundle names (see section 4).

### Demographics

- **`Adult_Dependents`** _(integer, 0–55, mean 1.89)_ — Number of adult dependents on the policy. Most customers have 2. Max of 55 is likely a data entry error.
- **`Child_Dependents`** _(float, 0–10, mean 0.15)_ — Number of child dependents. Has 4 missing values in train; fill with 0.
- **`Infant_Dependents`** _(integer, 0–9, mean 0.011)_ — Number of infant dependents. Rare — 98%+ of customers have 0.
- **`Estimated_Annual_Income`** _(float, $0–$1,836,758)_ — Customer's estimated gross annual income. Heavily right-skewed (max is 48× mean); apply `log1p` before use.
- **`Employment_Status`** _(string, 4 values)_ — Employment type: `Employed_FullTime` (83%), `Employed_PartTime`, `Self_Employed`, `Unemployed`. Use one-hot encoding.
- **`Region_Code`** _(string, 166 unique, 303 missing in train)_ — ISO-style country/region code. Top value is `PRT` (31%). High cardinality — use target or frequency encoding.

### Risk History

- **`Existing_Policyholder`** _(integer, 0/1)_ — 1 if the customer already held a policy with this insurer. Only 3.75% are existing holders.
- **`Previous_Claims_Filed`** _(integer, 0–26, mean 0.03)_ — Claims filed in prior policies. Over 97% have zero claims; useful mainly in composite risk features.
- **`Years_Without_Claims`** _(integer, 0–71, mean 0.17)_ — Consecutive years since last claim. Most customers are at 0, reflecting a predominantly new customer base.
- **`Previous_Policy_Duration_Months`** _(integer, 0–50, mean 2.66)_ — Length of the most recent prior policy in months. Reflects prior insurance experience.
- **`Policy_Cancelled_Post_Purchase`** _(integer, 0/1)_ — 1 if the customer cancelled a policy after buying it. **27.7% rate** — a strong churn/risk signal.

### Policy Preferences

- **`Deductible_Tier`** _(string, 4 values, 314 missing)_ — Chosen deductible level: `Tier_1_High_Ded` (79%), `Tier_2`, `Tier_3`, `Tier_4_Zero_Ded`. Higher tier = lower deductible = more premium. Use ordinal encoding.
- **`Payment_Schedule`** _(string, 3 values)_ — Billing frequency: `Monthly_EFT` (98.7%), `Quarterly`, `Annual`. Near-zero variance — **drop this column**.
- **`Vehicles_on_Policy`** _(integer, 0–8, mean 0.09)_ — Number of vehicles on the policy. 85%+ have 0; any vehicle strongly signals auto bundle preference.
- **`Custom_Riders_Requested`** _(integer, 0–5, mean 0.71)_ — Optional add-on riders requested. More riders correlate with premium bundle selection.
- **`Grace_Period_Extensions`** _(integer, 0–19, mean 1.02)_ — Payment deadline extensions granted. Average of 1; higher values signal financial stress.

### Sales & Underwriting

- **`Days_Since_Quote`** _(integer, 0–678, median 49)_ — Days between the initial quote and purchase. Fast buyers (<7 days) skew to `Basic_Health`; slow buyers (>200 days) to premium bundles.
- **`Underwriting_Processing_Days`** _(integer, 0–391, median 0)_ — Days for underwriting approval. Median is 0 (instant approval); a binary `Underwriting_Delayed` flag is more useful than the raw value.
- **`Policy_Amendments_Count`** _(integer, 0–18, mean 0.27)_ — Times the customer amended the policy before finalising. Higher counts signal indecision or complex needs.
- **`Acquisition_Channel`** _(string, 5 values, 666 missing)_ — How the customer was acquired: `Aggregator_Site` (60%), `Local_Broker`, `Direct_Website`, `Referral`, `Corporate_HR`.
- **`Broker_Agency_Type`** _(string, 2 values)_ — Brokerage type: `Urban_Boutique` (61%) or `National_Corporate` (39%). Encode as binary 0/1.
- **`Broker_ID`** _(float, 315 unique, 8,331 missing)_ — ID of the selling broker. **Top predictor** — brokers specialise in specific bundles (e.g. Broker 1.0 sells almost exclusively `Basic_Health`). Use target encoding.
- **`Employer_ID`** _(float, 309 unique, 57,396 missing = 94.3%)_ — Employer ID for group/corporate policies. Useless as a raw number; create a binary `Employer_ID_Present` flag to identify group-sponsored customers.

### Temporal

- **`Policy_Start_Year`** _(integer, 2015–2017)_ — Year the policy became active. Three-year window; mild predictive value as product mix shifted over time.
- **`Policy_Start_Month`** _(string, 12 month names)_ — Month stored as a full name (e.g. `"April"`). Must be mapped to integers 1–12, then cyclically encoded with sin/cos. December is most frequent (~13%).
- **`Policy_Start_Week`** _(integer, 1–53, mean 26.9)_ — ISO week number. Roughly uniform distribution. Apply cyclical sin/cos encoding.
- **`Policy_Start_Day`** _(integer, 1–31, mean 15.8)_ — Day of month. Roughly uniform; near-zero correlation with target — **drop this column**.

---

## 4. Target Variable Analysis

### Bundle Mapping

| ID  | Bundle Name          | Description                  |
| --- | -------------------- | ---------------------------- |
| 0   | Auto_Comprehensive   | Full vehicle coverage        |
| 1   | Auto_Liability_Basic | Basic liability for vehicles |
| 2   | Basic_Health         | Entry-level health plan      |
| 3   | Family_Comprehensive | Full family coverage         |
| 4   | Health_Dental_Vision | Health + dental + vision     |
| 5   | Home_Premium         | Premium homeowner policy     |
| 6   | Home_Standard        | Standard homeowner policy    |
| 7   | Premium_Health_Life  | Health + life combo          |
| 8   | Renter_Basic         | Basic renter's insurance     |
| 9   | Renter_Premium       | Premium renter's insurance   |

### Class Distribution (Train)

| Bundle               | Count      | Share     |
| -------------------- | ---------- | --------- |
| Basic_Health         | **36,136** | **59.4%** |
| Health_Dental_Vision | ~4,900     | ~8.1%     |
| Auto_Comprehensive   | ~4,400     | ~7.2%     |
| Family_Comprehensive | ~3,900     | ~6.4%     |
| Auto_Liability_Basic | ~3,500     | ~5.8%     |
| Premium_Health_Life  | ~2,900     | ~4.8%     |
| Home_Standard        | ~2,000     | ~3.3%     |
| Home_Premium         | ~1,500     | ~2.5%     |
| Renter_Basic         | ~900       | ~1.5%     |
| Renter_Premium       | ~700       | ~1.2%     |

### Critical Finding — Extreme Class Imbalance

`Basic_Health` dominates with **59.4%** of all training rows. The smallest class
(`Renter_Premium`) holds about **1.2%** — a **50× ratio** between the largest and smallest class.

**This is the most important problem to solve.** Macro F1 scores every class equally,
so a model that always predicts `Basic_Health` gets near-zero Macro F1 despite 59% raw accuracy.

Mandatory mitigation strategies:

- `class_weight='balanced'` in tree-based models
- Stratified K-Fold cross-validation
- SMOTE oversampling for minority classes
- Per-class probability threshold tuning at inference time

---

## 5. Missing Values

### Train Set

| Column                | Missing | Pct       | Recommended Strategy            |
| --------------------- | ------- | --------- | ------------------------------- |
| `Employer_ID`         | 57,396  | **94.3%** | Binary flag; drop raw value     |
| `Broker_ID`           | 8,331   | **13.7%** | Median fill + missing indicator |
| `Acquisition_Channel` | 666     | 1.1%      | Mode fill (`Aggregator_Site`)   |
| `Deductible_Tier`     | 314     | 0.5%      | Mode fill (`Tier_1_High_Ded`)   |
| `Region_Code`         | 303     | 0.5%      | Add `"Unknown"` category        |
| `Child_Dependents`    | 4       | 0.01%     | Fill with 0                     |

### Test Set

| Column                | Missing | Pct   |
| --------------------- | ------- | ----- |
| `Employer_ID`         | 14,350  | 94.3% |
| `Broker_ID`           | 2,077   | 13.7% |
| `Acquisition_Channel` | 143     | 0.9%  |
| `Deductible_Tier`     | 95      | 0.6%  |
| `Region_Code`         | 89      | 0.6%  |

Missing rates are **consistent between train and test** — no distribution shift in the missingness pattern itself.

### The Employer_ID Insight

At 94.3% missing, `Employer_ID` as a numeric feature is useless. However, the 5.7%
of rows that _have_ an Employer_ID identify **group/employer-sponsored policies** —
a meaningfully different customer segment. The missing indicator flag is a genuine
feature, not just imputation bookkeeping.

---

## 6. Demographics & Financials

### Estimated Annual Income

| Statistic | Value          |
| --------- | -------------- |
| Min       | $0             |
| 25th pct  | $25,178        |
| Median    | $34,883        |
| Mean      | $37,961        |
| 75th pct  | $48,048        |
| **Max**   | **$1,836,758** |

- Heavily **right-skewed** — the max is 48× the mean
- At least one extreme outlier above $1.8M
- Higher-income customers skew toward `Home_Premium`, `Premium_Health_Life`, and `Family_Comprehensive`; lower-income toward `Basic_Health` and `Renter_Basic`
- **Action:** Apply `log1p` transform or winsorise at the 99th percentile (~$110k)

### Dependents

| Feature             | Mean  | Median | Max |
| ------------------- | ----- | ------ | --- |
| `Adult_Dependents`  | 1.89  | 2      | 55  |
| `Child_Dependents`  | 0.15  | 0      | 10  |
| `Infant_Dependents` | 0.011 | 0      | 9   |

- Most customers have **2 adult dependents** and no children
- `Adult_Dependents` max of 55 is likely a data entry error
- `Has_Children` (child + infant > 0) separates family-oriented buyers from others
- `Family_Comprehensive` and `Health_Dental_Vision` show higher average child counts

### Employment Status

| Status            | Count  | Share     |
| ----------------- | ------ | --------- |
| Employed_FullTime | 50,382 | **82.8%** |
| Other 3 statuses  | 10,486 | 17.2%     |

Employment status is heavily concentrated. The 17.2% non-full-time workers are a distinct segment worth examining for bundle preference differences.

### Region Code

| Region            | Count  | Share     |
| ----------------- | ------ | --------- |
| PRT (Portugal)    | 18,863 | **31.1%** |
| GBR (UK)          | 7,417  | 12.2%     |
| FRA (France)      | 6,172  | 10.2%     |
| ESP (Spain)       | 5,067  | 8.3%      |
| DEU (Germany)     | 3,697  | 6.1%      |
| 161 other regions | 19,349 | 31.8%     |

- **166 unique region codes** — one-hot encoding is not viable
- Top 5 countries cover 68% of data
- Use **target encoding** or **frequency encoding** for this column

---

## 7. Customer History & Risk Profile

### Risk Metrics

| Feature                           | Mean  | Median | Max | Notable                            |
| --------------------------------- | ----- | ------ | --- | ---------------------------------- |
| `Previous_Claims_Filed`           | 0.029 | 0      | 26  | 97%+ have zero claims              |
| `Years_Without_Claims`            | 0.17  | 0      | 71  | Mostly new customers               |
| `Previous_Policy_Duration_Months` | 2.66  | 2      | 50  | Short prior policy history         |
| `Grace_Period_Extensions`         | 1.02  | 1      | 19  | Avg 1 extension — financial stress |
| `Policy_Amendments_Count`         | 0.27  | 0      | 18  | Most customers don't amend         |

### Key Risk Observations

**Almost nobody has filed a claim.** Mean = 0.029 means fewer than 3 in 100 customers
have any prior claim. The claims columns will have low individual predictive power
but contribute to composite risk features.

**Grace Period Extensions average 1.02** — the average customer has requested at
least one payment deadline extension. Higher grace extensions correlate with lower-tier bundles.

### Binary Risk Flags

| Flag                             | Rate      | Insight                                  |
| -------------------------------- | --------- | ---------------------------------------- |
| `Existing_Policyholder`          | **3.75%** | 96.25% are new customers                 |
| `Policy_Cancelled_Post_Purchase` | **27.7%** | 1 in 4 customers cancelled post-purchase |

The **27.7% cancellation rate** is the most striking flag. Over 1 in 4 customers
cancelled a policy shortly after buying. This is a strong behavioural signal —
engineer a `High_Churn_Risk` composite feature.

---

## 8. Policy Details & Preferences

### Deductible Tier

| Tier            | Count  | Share     |
| --------------- | ------ | --------- |
| Tier_1_High_Ded | 47,688 | **78.5%** |
| Tier_2          | ~7,000 | ~11.5%    |
| Tier_3          | ~3,500 | ~5.8%     |
| Tier_4_Zero_Ded | ~2,366 | ~3.9%     |

Deductible tier directly correlates with bundle type — high-deductible customers
skew toward `Basic_Health`; zero-deductible toward `Premium_Health_Life` and `Home_Premium`.

### Payment Schedule

| Schedule    | Count  | Share     |
| ----------- | ------ | --------- |
| Monthly_EFT | 60,092 | **98.7%** |
| Others      | 776    | 1.3%      |

**Near-zero variance. Drop this column.** It contributes nothing to prediction.

### Vehicles on Policy

| Statistic | Value |
| --------- | ----- |
| Mean      | 0.086 |
| Median    | 0     |
| Max       | 8     |

85%+ of customers have 0 vehicles. Customers with vehicles (> 0) strongly signal auto bundle preference.

### Custom Riders Requested

| Statistic | Value |
| --------- | ----- |
| Mean      | 0.71  |
| Median    | 0     |
| Max       | 5     |

Premium bundle buyers request more add-ons. Useful complexity indicator.

---

## 9. Sales, Brokers & Acquisition

### Broker_ID — Strongest Categorical Predictor

`Broker_ID` has **315 unique values** but is extremely concentrated:

| Broker     | Records | Share     |
| ---------- | ------- | --------- |
| 9.0        | 20,065  | **33.0%** |
| 240.0      | 9,317   | **15.3%** |
| 14.0       | 2,368   | 3.9%      |
| 7.0        | 2,307   | 3.8%      |
| 250.0      | 2,076   | 3.4%      |
| All others | ~24,735 | 40.6%     |

Brokers 9 and 240 together account for **48.3%** of all training data.

**Brokers specialise in specific bundles:**

| Broker | Mean Bundle ID | Std      | Interpretation                   |
| ------ | -------------- | -------- | -------------------------------- |
| 1.0    | 2.001          | **0.21** | Exclusively sells `Basic_Health` |
| 6.0    | 2.278          | 0.71     | Mostly Basic_Health              |
| 28.0   | 2.423          | 0.88     | Low-tier specialist              |
| 240.0  | 2.726          | 1.14     | Mixed but Health-skewed          |
| 9.0    | 2.834          | 1.41     | Broad portfolio                  |
| 241.0  | 3.502          | 1.85     | Premium-leaning                  |
| 250.0  | 3.822          | **1.98** | Premium bundle specialist        |

Broker 1.0's std of **0.21** (on a 0–9 scale) is extraordinary — it almost never
sells anything other than `Basic_Health`. **Target-encode `Broker_ID`** — this will
be among the most predictive features in the model.

### Broker Agency Type

Only **2 unique values**: `Urban_Boutique` (60.5%) and `National_Corporate` (39.5%).
Boutique agencies tend toward different bundle mixes than corporate ones.

### Acquisition Channel

| Channel         | Count  | Share     |
| --------------- | ------ | --------- |
| Aggregator_Site | 36,306 | **60.3%** |
| Local_Broker    | ~9,000 | ~14.9%    |
| Direct_Website  | ~7,500 | ~12.5%    |
| Referral        | ~4,700 | ~7.8%     |
| Corporate_HR    | ~2,700 | ~4.5%     |

`Corporate_HR` channel likely overlaps with customers who have an `Employer_ID` — both signal group policies.

### Underwriting Processing Days

| Statistic | Value   |
| --------- | ------- |
| Mean      | 0.72    |
| Median    | **0**   |
| 75th pct  | 0       |
| Max       | **391** |

Most policies are instant-approved. A binary `Underwriting_Delayed` flag (> 0 days) is far more useful than the raw value.

### Days Since Quote

| Statistic | Value |
| --------- | ----- |
| Min       | 0     |
| Median    | 49    |
| Mean      | 79.8  |
| Max       | 678   |

Wide spread — some customers decide same-day, others take nearly 2 years. Decision speed correlates with bundle complexity:

- **Fast buyers (< 7 days):** likely `Basic_Health` (simple, standard)
- **Slow buyers (> 200 days):** likely premium or complex bundles

### Policy Amendments Count

| Statistic | Value |
| --------- | ----- |
| Mean      | 0.27  |
| Median    | 0     |
| Max       | 18    |

High amendment counts indicate complex needs or indecision. Combine with `Days_Since_Quote` for an **indecision interaction feature**.

---

## 10. Temporal Patterns

### Policy Start Year

Data covers **2015–2017** only — a 3-year window. Year has some predictive value as the product mix may have shifted between years.

### Policy Start Month

All 12 months are represented. December is the most frequent (7,930 records, ~13%).
Month is stored as a **string** — must be mapped to an integer before any encoding.

```python
MONTH_ORDER = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
```

Apply **cyclical sin/cos encoding** to prevent the model treating December (12) and January (1) as maximally distant.

### Policy Start Week & Day

- `Policy_Start_Week`: 1–53, mean 26.9 — roughly uniform across the year
- `Policy_Start_Day`: 1–31, mean 15.8 — roughly uniform

Both have **near-zero correlation** with the target. Drop `Policy_Start_Day`; apply cyclical encoding to week.

---

## 11. Correlation Analysis — All Features

The full heatmap (`correlation_heatmap_full.png`) label-encodes all categorical
columns and computes Pearson correlation across every feature and the target.

### Top Features by Absolute Correlation with Target

| Rank | Feature                   | Strength    | Note                                  |
| ---- | ------------------------- | ----------- | ------------------------------------- |
| 1    | `Broker_ID`               | High        | Broker specialisation                 |
| 2    | `Acquisition_Channel`     | High        | Channel-bundle affinity               |
| 3    | `Deductible_Tier`         | Medium-High | Tier selection mirrors bundle tier    |
| 4    | `Region_Code`             | Medium      | Geographic preference patterns        |
| 5    | `Estimated_Annual_Income` | Medium      | Income drives premium vs basic choice |
| 6    | `Broker_Agency_Type`      | Medium      | Boutique vs corporate mix             |
| 7    | `Custom_Riders_Requested` | Medium      | More riders signal premium bundles    |
| 8    | `Employment_Status`       | Low-Medium  | Full-time employment skews to Basic   |
| 9    | `Grace_Period_Extensions` | Low         | Financial stress signal               |
| 10   | `Adult_Dependents`        | Low         | Family size                           |
| 11   | `Vehicles_on_Policy`      | Low         | Auto vs non-auto signal               |
| 12   | `Days_Since_Quote`        | Low         | Decision speed                        |
| 13+  | Temporal features         | Near-zero   | Week, Day — minimal signal            |

### Notable Inter-Feature Correlations

| Pair                                      | Observation                                |
| ----------------------------------------- | ------------------------------------------ |
| `Broker_ID` vs `Acquisition_Channel`      | Brokers operate through specific channels  |
| `Income` vs `Deductible_Tier`             | Higher income, lower deductible preference |
| `Custom_Riders` vs `Policy_Amendments`    | Complex customers modify more              |
| `Grace_Extensions` vs `Policy_Cancelled`  | Financial stress leads to cancellation     |
| `Underwriting_Days` vs `Days_Since_Quote` | Delayed underwriting extends quote period  |
| `Payment_Schedule` vs Target              | Near-zero, confirms it should be dropped   |

---

## 12. Test Set Analysis

The test set has **15,218 rows** and the same 28 feature columns as train (no target).
Section 14 of the notebook (`eda.ipynb`) produces 7 charts that examine the test
population in detail. Key findings are summarised below.

### 12.1 Missing Values

Missing rates are virtually identical across both splits — the imputation strategy
designed on train transfers cleanly to test without adjustment.

| Column                | Train missing | Test missing | Delta  |
| --------------------- | ------------- | ------------ | ------ |
| `Employer_ID`         | 94.30%        | 94.30%       | 0.00%  |
| `Broker_ID`           | 13.69%        | 13.65%       | −0.04% |
| `Acquisition_Channel` | 1.09%         | 0.94%        | −0.15% |
| `Deductible_Tier`     | 0.52%         | 0.62%        | +0.10% |
| `Region_Code`         | 0.50%         | 0.58%        | +0.08% |

No column shows a meaningful gap — safe to apply the same filling strategy to both sets.

### 12.2 Categorical Features

All five categorical features show near-identical distributions in train and test:

- **Employment Status:** `Employed_FullTime` dominates at ~83% in both splits. The share of `Self_Employed`, `Employed_PartTime`, and `Unemployed` customers is stable.
- **Broker Agency Type:** `Urban_Boutique` is ~61% and `National_Corporate` ~39% in both — no shift.
- **Acquisition Channel:** `Aggregator_Site` holds ~60% in both. `Corporate_HR` and `Referral` shares are consistent.
- **Deductible Tier:** `Tier_1_High_Ded` at ~79% in both. All four tier proportions match within noise.
- **Payment Schedule:** Remains 98.7% `Monthly_EFT` in both — reinforces the decision to drop this column.

### 12.3 Top Regions

The regional distribution in test mirrors train almost exactly. `PRT` is the dominant
region at ~31% in both. The top 15 regions maintain the same relative ordering and
share. No new region codes appear in test that are unseen in train.

### 12.4 Top Brokers

Broker distribution is consistent. Broker 9 (33%) and Broker 240 (15%) retain the
same dominance in the test set. No significant broker appears in test that is absent
from train, which is critical because target-encoding `Broker_ID` requires every
test value to have been seen during training.

### 12.5 Numerical Features — KS Test Results

Kolmogorov-Smirnov tests were run on all 17 numerical features. **No feature shows
a statistically significant distribution shift** (all KS p-values > 0.05).

The features with the largest KS statistics (though still non-significant) are:

- `Underwriting_Processing_Days` — slightly different tail in test, but median still 0
- `Days_Since_Quote` — minor skew difference, central tendency identical
- `Estimated_Annual_Income` — consistent shape; outlier at $1.8M may differ slightly

Features with near-zero KS statistics (perfectly aligned):

- `Policy_Start_Week`, `Policy_Start_Day`, `Adult_Dependents`, `Infant_Dependents`

### 12.6 Temporal Distributions

- **Year:** Both train and test cover 2015–2017. The year proportions (roughly 20% / 45% / 35% for 2015/2016/2017) are identical.
- **Month:** All 12 months are present in test. The slight December peak observed in train (~13%) is replicated in test. No seasonal shift.
- **Week:** The roughly uniform week distribution in train is reproduced in test.

### 12.7 Overall Conclusion — Test Set

> The test set is a **clean random stratified hold-out** from the same population as train.
> There is no temporal leakage, geographic shift, or broker composition change.
> All preprocessing pipelines, encoders, and imputers fitted on train can be applied
> to test without modification. Standard cross-validation scores on train are reliable
> predictors of test performance.

---

## 13. Key Findings Summary

| #   | Finding                                            | Impact on Modelling                               |
| --- | -------------------------------------------------- | ------------------------------------------------- |
| 1   | `Basic_Health` = 59.4% of data                     | Must use class balancing; Macro F1 is the target  |
| 2   | `Employer_ID` is 94.3% missing                     | Use presence/absence flag only                    |
| 3   | Broker 9 handles 33% of all records                | `Broker_ID` is top predictor; target-encode it    |
| 4   | Broker 1.0 sells only `Basic_Health` (std = 0.21)  | Broker specialisation is near-deterministic       |
| 5   | `Payment_Schedule` is 98.7% one value              | Drop — no variance, no signal                     |
| 6   | Income max = $1.8M (48× mean)                      | Log-transform or winsorise                        |
| 7   | Underwriting median = 0 days                       | Binary delayed flag more useful than raw value    |
| 8   | 27.7% cancellation rate                            | Strong churn/risk behavioural signal              |
| 9   | 97%+ have zero prior claims                        | Claims columns only useful in composites          |
| 10  | PRT = 31% of all records                           | Target-encode `Region_Code`; don't one-hot        |
| 11  | `Policy_Start_Month` is a string                   | Must map to int then cyclical-encode              |
| 12  | No train/test distribution shift (all KS p > 0.05) | Standard cross-validation is safe                 |
| 13  | Broker composition identical in train and test     | Target-encoding Broker_ID is safe — no cold-start |
| 14  | All region codes in test exist in train            | No unseen-category problem for Region_Code        |

---

## 14. Feature Engineering Recommendations

### New Features to Create

| Feature                 | Formula / Logic                            | Rationale                        |
| ----------------------- | ------------------------------------------ | -------------------------------- |
| `Total_Dependents`      | adult + child.fillna(0) + infant           | Overall household size           |
| `Has_Children`          | (child + infant) > 0 as 0/1                | Family vs non-family customer    |
| `Employer_ID_Missing`   | isna(Employer_ID) as 0/1                   | Group vs individual policy       |
| `Broker_ID_Missing`     | isna(Broker_ID) as 0/1                     | Data completeness signal         |
| `log_Income`            | log1p(Estimated_Annual_Income)             | Handle extreme right skew        |
| `Income_Per_Dependent`  | income / (Total_Dependents + 1)            | Per-person affordability         |
| `Risk_Score`            | claims − years_clean + grace_extensions    | Composite financial stress       |
| `Underwriting_Delayed`  | Underwriting_Processing_Days > 0 as 0/1    | Complex risk flag                |
| `Fast_Buyer`            | Days_Since_Quote < 7 as 0/1                | Quick-decision signal            |
| `Indecision_Score`      | Days_Since_Quote × Policy_Amendments_Count | Deliberation complexity          |
| `Policy_Complexity`     | Custom_Riders + Policy_Amendments          | Overall policy complexity        |
| `Month_Num`             | MONTH_ORDER[Policy_Start_Month]            | Intermediate for cyclical encode |
| `Month_Sin/Cos`         | sin/cos(2pi × Month_Num / 12)              | Seasonal cyclical encoding       |
| `Week_Sin/Cos`          | sin/cos(2pi × Policy_Start_Week / 52)      | Seasonal cyclical encoding       |
| `Broker_ID_TargetEnc`   | mean(target) per Broker_ID                 | Broker specialisation            |
| `Region_TargetEnc`      | mean(target) per Region_Code               | Geographic preference            |
| `Employer_ID_TargetEnc` | mean(target) per Employer_ID (non-null)    | Group policy patterns            |

### Encoding Strategy for Categoricals

| Column                | Strategy                             | Reason                       |
| --------------------- | ------------------------------------ | ---------------------------- |
| `Broker_ID`           | Target encoding                      | 315 unique, strong signal    |
| `Region_Code`         | Target or frequency encoding         | 166 unique, high-cardinality |
| `Employer_ID`         | Binary flag + target enc on non-null | 94% missing                  |
| `Employment_Status`   | One-hot (4 values)                   | Low cardinality              |
| `Broker_Agency_Type`  | Binary encode (2 values)             | Binary already               |
| `Deductible_Tier`     | Ordinal encode (Tier 1 to 4)         | Natural order                |
| `Acquisition_Channel` | One-hot (5 values)                   | Low cardinality              |
| `Payment_Schedule`    | **Drop**                             | 98.7% single value           |

---

## 15. Columns to Drop

| Column              | Reason                                     |
| ------------------- | ------------------------------------------ |
| `User_ID`           | Identifier — no predictive value           |
| `Payment_Schedule`  | 98.7% `Monthly_EFT` — near-zero variance   |
| `Employer_ID` (raw) | 94.3% missing — replaced by binary flag    |
| `Policy_Start_Day`  | Near-zero correlation with target          |
| `Bundle_Name`       | Derived from target — data leakage if kept |

---

## 16. Model Strategy

### Recommended Models

Given the constraints: **50 MB zip · 1 GB RAM · 1 CPU core · 120 s total · 10 s predict latency**

| Model         | Pros                                | Cons                              |
| ------------- | ----------------------------------- | --------------------------------- |
| **LightGBM**  | Fast, small, handles categoricals   | Needs hypertuning                 |
| **CatBoost**  | Native categorical handling, robust | Larger model files                |
| **XGBoost**   | Battle-tested, good accuracy        | Heavier files                     |
| Random Forest | Simple, interpretable               | Too slow and large for 10 s limit |
| Deep Learning | Can model interactions              | Model size penalty kills score    |

**Recommendation: LightGBM with `class_weight='balanced'`** as the primary model.
Target-encode high-cardinality categoricals before passing to the model.

### Cross-Validation Setup

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Stratification is mandatory to ensure all 10 classes appear in every fold,
especially for the minority classes (`Renter_Basic`, `Renter_Premium` at < 2%).

### Imbalance Handling Priority

1. `class_weight='balanced'` — free, no overfitting risk
2. Stratified K-Fold — always
3. SMOTE on training fold only — after split to prevent leakage
4. Post-training threshold optimisation per class — to maximise Macro F1

### Explainability (Bonus Points)

The scoring criteria award 10 points for explainability (SHAP, feature importances).
Use SHAP with LightGBM:

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

## File Structure

```text
DataQuest/
├── Data/
│   ├── train.csv
│   ├── test.csv
│   ├── train_clean.csv                 <- Cleaned + engineered training features
│   └── test_clean.csv                  <- Cleaned + engineered test features
├── notebook/
│   ├── eda.ipynb                       <- Full EDA + test set analysis
│   ├── cleaning.ipynb                  <- Full cleaning pipeline
│   ├── modelling.ipynb                 <- CV benchmarking & model selection
│   ├── submit.ipynb                    <- v1 submission builder
│   ├── submit_v2.ipynb                 <- v2 submission builder (best latency config)
│   └── submit_v7.ipynb                 <- v7 submission builder (best overall)
├── submission_v7/
│   ├── solution.py                     <- Judge-ready inference script
│   ├── model.pkl                       <- Trained LightGBM + encoding maps
│   └── requirements.txt
├── submission_v7.zip                   <- Current best submission
├── test/
│   └── test_latency.py                 <- Local judge simulator
├── venv/                               <- Python virtual environment
├── DataQuest-Brief-Document.pdf
└── README.md                           <- This file
```

---

## 17. Data Cleaning Pipeline

Implemented in `notebook/cleaning.ipynb`. Produces `Data/train_clean.csv` (60,868 × 45) and `Data/test_clean.csv` (15,218 × 44).

### Columns Dropped

| Column | Reason |
| --- | --- |
| `User_ID` | Identifier — no predictive value |
| `Payment_Schedule` | 98.7% `Monthly_EFT` — zero variance |
| `Policy_Start_Day` | Near-zero correlation with target |

### Missing Value Fills

| Column | Strategy |
| --- | --- |
| `Child_Dependents` | Fill with 0 |
| `Deductible_Tier` | Fill with `Tier_1_High_Ded` (mode) |
| `Acquisition_Channel` | Fill with `Aggregator_Site` (mode) |
| `Region_Code` | Fill with `"UNKNOWN"` |
| `Broker_ID` / `Employer_ID` | Left as NaN — handled by encoding |

### Discovered Column Value Corrections

> The README originally listed incorrect category names. Actual values found during cleaning:

| Column | README said | Actual value in data |
| --- | --- | --- |
| `Deductible_Tier` | `Tier_2`, `Tier_3` | `Tier_2_Mid_Ded`, `Tier_3_Low_Ded` |
| `Employment_Status` | `Employed_PartTime` | `Contractor` |
| `Acquisition_Channel` | `Referral`, `Corporate_HR` | `Affiliate_Group`, `Corporate_Partner` |

### Engineered Features (17 new columns)

| Feature | Formula |
| --- | --- |
| `Total_Dependents` | `Adult + Child.fillna(0) + Infant` |
| `Has_Children` | `(Child + Infant) > 0` |
| `Employer_ID_Present` | `Employer_ID.notna()` |
| `Broker_ID_Missing` | `Broker_ID.isna()` |
| `log_Income` | `log1p(Estimated_Annual_Income)` |
| `log_Income_Per_Dependent` | `log1p(Income / (Total_Dependents + 1))` |
| `Risk_Score` | `Claims − Years_Clean + Grace_Extensions` |
| `Underwriting_Delayed` | `Underwriting_Processing_Days > 0` |
| `Fast_Buyer` | `Days_Since_Quote < 7` |
| `Indecision_Score` | `Days_Since_Quote × Policy_Amendments_Count` |
| `Policy_Complexity` | `Custom_Riders + Policy_Amendments` |
| `Month_Num` | `MONTH_ORDER[Policy_Start_Month]` |
| `Month_Sin / Month_Cos` | `sin/cos(2π × Month_Num / 12)` |
| `Week_Sin / Week_Cos` | `sin/cos(2π × Policy_Start_Week / 52)` |

### Encoding Applied

| Column | Method |
| --- | --- |
| `Deductible_Tier` | Ordinal (1–4) |
| `Broker_Agency_Type` | Binary (`National_Corporate` = 1) |
| `Employment_Status` | One-hot (3 dummies, drop `Contractor`) |
| `Acquisition_Channel` | One-hot (4 dummies, drop `Affiliate_Group`) |
| `Broker_ID` | Smoothed target encoding (k=300) |
| `Region_Code` | Smoothed target encoding (k=300) |
| `Employer_ID` | Smoothed target encoding (k=300) |

---

## 18. Submission History

All submissions used LightGBM multiclass on 15,218 test rows. Scoring formula:
`score = Macro_F1 × max(0.5, 1−size/200) × max(0.5, 1−latency/10)`

| Ver | Key Change | F1 | Latency | Score | Judge Factor |
| --- | --- | --- | --- | --- | --- |
| v1 | 150 trees, threshold tuning | 0.4719 | 2.83s | 0.3331 | 6.9× |
| v2 | 75 trees, no threshold | 0.5176 | 1.07s | 0.4583 | 5.6× |
| v3 | 50 trees (latency test) | 0.4487 | 0.47s | 0.4253 | 3.6× |
| v4 | 150 shallow trees depth=4 | 0.4798 | 1.23s | 0.4188 | 9.1× |
| v5 | Full-train encoding fix | 0.5217 | 1.03s | 0.4640 | 9.9× |
| v7 | All improvements combined | **0.5577** | 1.18s | **0.4875** | 10.0× |

### Key Lessons Learned

**1. Latency/size optimisation backfired (v3, v4)**
Reducing trees from 75→50 or changing to shallow trees (depth=4) caused F1 to drop more than the latency penalty gain compensated. The judge server speed varies wildly (3.6×–10× local) making latency unpredictable.

**2. Judge factor is highly variable**
Local latency × judge factor = actual latency, but the judge factor ranged from 3.56× to 9.99× across submissions. Local benchmarks are unreliable for latency prediction.

**3. OOF vs full-train encoding inconsistency (v5 fix)**
`train_clean.csv` used OOF encoding for `Broker_ID_TargetEnc`, `Region_TargetEnc`, `Employer_ID_TargetEnc`. But `solution.py` applied full-train encoding at inference. The model saw slightly different values during training vs inference — fixing this gave +0.004 in actual F1.

**4. `class_weight='balanced'` with SMOTE is redundant**
When SMOTE is already balancing classes, adding `class_weight='balanced'` double-penalises the majority. Setting `class_weight=None` after SMOTE gave +0.022 OOF F1.

**5. Threshold tuning overfits**
Per-class thresholds tuned on OOF validation folds (v1) inflated training F1 to 0.74 but hurt actual score — the thresholds were memorising fold-specific noise.

---

## 19. Best Model Configuration (v7)

### Architecture

- **Model:** LightGBM multiclass (10 classes)
- **Trees:** 75 per class (750 total)
- **Hyperparameters:** `num_leaves=95`, `learning_rate=0.07`, `max_depth=7`, `min_child_samples=20`, `reg_alpha=0.1`, `reg_lambda=0.5`, `subsample=0.8`, `colsample_bytree=0.8`
- **Class weight:** `None` (SMOTE handles imbalance)

### Feature Set (62 features)

All 44 features from `train_clean.csv`, **replacing** `Broker_ID_TargetEnc` and `Region_TargetEnc` with **multi-class target encoding** (10 features each):

| Feature group | Count | Description |
| --- | --- | --- |
| Original numeric features | 32 | Policy, risk, temporal, financial |
| One-hot dummies | 7 | Employment (3), Acquisition (4) |
| Binary flags | 3 | `Agency_National`, `Employer_ID_Present`, `Broker_ID_Missing` |
| `Employer_ID_TargetEnc` | 1 | Smoothed mean target per employer |
| `Broker_C0`–`Broker_C9` | 10 | P(class=k \| broker) for k=0..9 |
| `Region_C0`–`Region_C9` | 10 | P(class=k \| region) for k=0..9 |

### Multi-class Target Encoding

Instead of a single number (mean target), `Broker_ID` and `Region_Code` are each encoded as **10 probability features** — one per class. This gives the model the full specialisation distribution, not just the average:

```python
# For each class k and each broker b:
# P(bundle=k | broker=b) = (n_bk * mean_bk + k * prior_k) / (n_bk + k)
# where prior_k = global P(class=k), k=300 (smoothing)
```

### Resampling Strategy

Applied **before** each training fold (never on validation data):

| Step | Target | Classes | Method |
| --- | --- | --- | --- |
| 1 | 200 samples | C8, C9 (5–6 real samples) | `RandomOverSampler` |
| 2 | 600 samples | C8, C9 | `SMOTE(k_neighbors=3)` |
| 3 | 3000 samples | C0, C5, C6 | `SMOTE(k_neighbors=5)` |

### OOF Results (5-fold)

| Class | Bundle | OOF F1 |
| --- | --- | --- |
| C2 | Basic_Health | 0.846 |
| C1 | Auto_Liability_Basic | 0.674 |
| C9 | Renter_Premium | 1.000 |
| C5 | Home_Premium | 0.715 |
| C7 | Premium_Health_Life | 0.648 |
| C6 | Home_Standard | 0.614 |
| C0 | Auto_Comprehensive | 0.548 |
| C4 | Health_Dental_Vision | 0.544 |
| C8 | Renter_Basic | 0.421 |
| C3 | Family_Comprehensive | 0.408 |
| **Macro** | | **0.6418** |

### File Sizes & Latency

| Metric | Value |
| --- | --- |
| `model.pkl` | 1.86 MB |
| ZIP total | 1.86 MB |
| Size penalty | 0.9907 |
| Local predict time | ~0.12s |
| Judge predict time | ~1.18s (10× factor) |
| Latency penalty | ~0.882 |
| **Best actual score** | **0.4875** |
