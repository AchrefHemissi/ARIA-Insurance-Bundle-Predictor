# Data Directory

> 60,868 training rows · 15,218 test rows · 10 insurance bundle classes

---

## Layout

```
Data/
├── raw/
│   ├── train.csv       ← 60,868 × 29 (28 features + target)
│   └── test.csv        ← 15,218 × 28
└── processed/
    ├── train_clean.csv ← 60,868 × 45 (output of cleaning.ipynb)
    └── test_clean.csv  ← 15,218 × 44
```

The v7 model adds 20 multi-class target-encoded features and drops 2 at train time → **62 final features**.

---

## Target Classes

| ID | Bundle | Count | Share |
|---:|--------|------:|------:|
| 0 | Auto_Comprehensive | 823 | 1.4% |
| 1 | Auto_Liability_Basic | 1,625 | 2.7% |
| 2 | Basic_Health | 36,136 | **59.4%** |
| 3 | Family_Comprehensive | 4,831 | 7.9% |
| 4 | Health_Dental_Vision | 13,958 | 22.9% |
| 5 | Home_Premium | 479 | 0.8% |
| 6 | Home_Standard | 719 | 1.2% |
| 7 | Premium_Health_Life | 2,286 | 3.8% |
| 8 | Renter_Basic | 6 | <0.01% |
| 9 | Renter_Premium | 5 | <0.01% |

---

## Key Columns

| Column | Type | Notes |
|--------|------|-------|
| User_ID | int | Unique identifier (submission key) |
| Age | float | Customer age; has missing values |
| Income | float | Annual income; has missing values |
| Broker_ID | int | 20 unique brokers — strong class signal |
| Region_Code | int | 5 regions |
| Number_of_Existing_Policies | int | Active policy count |
| Purchased_Coverage_Bundle | int 0–9 | **Target** (train only) |

Full column details and EDA → `notebook/data_visualization.ipynb`
