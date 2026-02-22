"""retrain.py — Retrain the LightGBM insurance-bundle model.

Usage (local):
    python retrain/retrain.py --train data/train.csv --out backend/model.pkl

Usage (CI):
    Triggered via .github/workflows/retrain.yml with workflow_dispatch.
    Expects TRAIN_CSV_PATH env-var or --train argument.
"""
import argparse
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CLASSES = 10
SEED = 42

MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}
DEDUCTIBLE_MAP = {
    "Tier_1_High_Ded": 1, "Tier_2_Mid_Ded": 2, "Tier_3_Low_Ded": 3, "Tier_4_Zero_Ded": 4,
}

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 95,
    "learning_rate": 0.07,
    "reg_lambda": 0.5,
    "n_estimators": 800,
    "verbose": -1,
    "seed": SEED,
}


# ---------------------------------------------------------------------------
# Preprocessing (mirrors backend/app/services/preprocessor.py)
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering — mirrors solution.py preprocessing."""
    user_ids = df["User_ID"].copy() if "User_ID" in df.columns else None

    drop_cols = ["User_ID", "Payment_Schedule", "Policy_Start_Day", "Purchased_Coverage_Bundle"]
    target = df["Purchased_Coverage_Bundle"].copy() if "Purchased_Coverage_Bundle" in df.columns else None
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["Child_Dependents"] = df["Child_Dependents"].fillna(0)
    df["Deductible_Tier"] = df["Deductible_Tier"].fillna("Tier_1_High_Ded")
    df["Acquisition_Channel"] = df["Acquisition_Channel"].fillna("Aggregator_Site")
    df["Region_Code"] = df["Region_Code"].fillna("UNKNOWN")

    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)

    df["Total_Dependents"] = df["Adult_Dependents"] + df["Child_Dependents"].fillna(0) + df["Infant_Dependents"]
    df["Has_Children"] = ((df["Child_Dependents"].fillna(0) + df["Infant_Dependents"]) > 0).astype(int)
    df["Employer_ID_Present"] = df["Employer_ID"].notna().astype(int) if "Employer_ID" in df.columns else 0
    df["Broker_ID_Missing"] = df["Broker_ID"].isna().astype(int) if "Broker_ID" in df.columns else 0
    df["Underwriting_Delayed"] = (df["Underwriting_Processing_Days"] > 0).astype(int)
    df["Fast_Buyer"] = (df["Days_Since_Quote"] < 7).astype(int)
    income = df["Estimated_Annual_Income"].clip(lower=0)
    df["log_Income"] = np.log1p(income)
    df["log_Income_Per_Dependent"] = np.log1p(income / (df["Total_Dependents"] + 1))
    df["Risk_Score"] = df["Previous_Claims_Filed"] - df["Years_Without_Claims"] + df["Grace_Period_Extensions"]
    df["Indecision_Score"] = df["Days_Since_Quote"] * df["Policy_Amendments_Count"]
    df["Policy_Complexity"] = df["Custom_Riders_Requested"] + df["Policy_Amendments_Count"]

    df["Month_Num"] = df["Policy_Start_Month"].map(MONTH_ORDER).fillna(1).astype(int)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    df["Week_Sin"] = np.sin(2 * np.pi * df["Policy_Start_Week"] / 52)
    df["Week_Cos"] = np.cos(2 * np.pi * df["Policy_Start_Week"] / 52)

    df["Deductible_Tier_Ord"] = df["Deductible_Tier"].map(DEDUCTIBLE_MAP).fillna(1).astype(int)
    df["Agency_National"] = (
        (df["Broker_Agency_Type"] == "National_Corporate").astype(int)
        if "Broker_Agency_Type" in df.columns
        else 0
    )

    emp = df["Employment_Status"] if "Employment_Status" in df.columns else pd.Series("Employed_FullTime", index=df.index)
    df["Emp_Employed_FullTime"] = (emp == "Employed_FullTime").astype(int)
    df["Emp_Self_Employed"] = (emp == "Self_Employed").astype(int)
    df["Emp_Unemployed"] = (emp == "Unemployed").astype(int)

    acq = df["Acquisition_Channel"] if "Acquisition_Channel" in df.columns else pd.Series("Aggregator_Site", index=df.index)
    df["Acq_Aggregator_Site"] = (acq == "Aggregator_Site").astype(int)
    df["Acq_Corporate_Partner"] = (acq == "Corporate_Partner").astype(int)
    df["Acq_Direct_Website"] = (acq == "Direct_Website").astype(int)
    df["Acq_Local_Broker"] = (acq == "Local_Broker").astype(int)

    if user_ids is not None:
        df["User_ID"] = user_ids.values
    return df, target


# ---------------------------------------------------------------------------
# Target encoding helpers
# ---------------------------------------------------------------------------
def _target_encode_single(train_col: pd.Series, target: pd.Series, smoothing: int = 10) -> dict:
    """Simple smoothed target-encoding map for a single categorical column."""
    global_mean = target.mean()
    stats = target.groupby(train_col).agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    return smooth.to_dict(), global_mean


def _multi_class_encode(train_col: pd.Series, target: pd.Series, n_classes: int) -> tuple[dict, list]:
    """Multi-class target encoding: one map per class."""
    multi = {}
    global_prior = []
    for cls in range(n_classes):
        binary = (target == cls).astype(int)
        enc_map, _ = _target_encode_single(train_col, binary)
        multi[cls] = enc_map
        global_prior.append(float(binary.mean()))
    return multi, global_prior


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(train_csv: str, output_path: str) -> None:
    """Full training pipeline: preprocess → encode → fit LightGBM → save model.pkl."""
    print(f"[retrain] Loading training data from {train_csv}")
    raw = pd.read_csv(train_csv)
    print(f"[retrain] Raw shape: {raw.shape}")

    df, target = preprocess(raw)
    assert target is not None, "Training CSV must contain 'Purchased_Coverage_Bundle' column"

    # ── Target encoding ──────────────────────────────────────────────────
    employer_enc_map, global_mean = _target_encode_single(
        df["Employer_ID"].fillna(-1) if "Employer_ID" in df.columns else pd.Series(-1, index=df.index),
        target,
    )

    broker_multi, global_prior = _multi_class_encode(
        df["Broker_ID"].fillna(-1) if "Broker_ID" in df.columns else pd.Series(-1, index=df.index),
        target,
        N_CLASSES,
    )

    region_multi, _ = _multi_class_encode(
        df["Region_Code"].fillna("UNKNOWN") if "Region_Code" in df.columns else pd.Series("UNKNOWN", index=df.index),
        target,
        N_CLASSES,
    )

    # Apply encodings to features
    if "Employer_ID" in df.columns:
        df["Employer_ID_TargetEnc"] = df["Employer_ID"].map(employer_enc_map).fillna(global_mean)
    else:
        df["Employer_ID_TargetEnc"] = global_mean

    broker_raw = df["Broker_ID"].fillna(-1) if "Broker_ID" in df.columns else pd.Series(-1, index=df.index)
    for cls in range(N_CLASSES):
        df[f"Broker_C{cls}"] = broker_raw.map(broker_multi[cls]).fillna(global_prior[cls])

    region_raw = df["Region_Code"].fillna("UNKNOWN") if "Region_Code" in df.columns else pd.Series("UNKNOWN", index=df.index)
    for cls in range(N_CLASSES):
        df[f"Region_C{cls}"] = region_raw.map(region_multi[cls]).fillna(global_prior[cls])

    # Drop non-numeric / id columns before training
    drop_for_train = ["User_ID", "Employer_ID", "Broker_ID", "Region_Code",
                      "Broker_Agency_Type", "Employment_Status", "Acquisition_Channel",
                      "Deductible_Tier", "Policy_Start_Month"]
    df = df.drop(columns=[c for c in drop_for_train if c in df.columns], errors="ignore")

    feature_cols = [c for c in df.columns if c not in ("User_ID",)]
    X = df[feature_cols].values
    y = target.values
    print(f"[retrain] Feature matrix: {X.shape}, target classes: {np.unique(y)}")

    # ── Train LightGBM ───────────────────────────────────────────────────
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
    booster = lgb.train(
        {k: v for k, v in LGB_PARAMS.items() if k != "n_estimators"},
        dtrain,
        num_boost_round=LGB_PARAMS["n_estimators"],
    )

    # ── Package & save ───────────────────────────────────────────────────
    payload = {
        "model_str": booster.model_to_string(),
        "feature_cols": feature_cols,
        "employer_enc_map": employer_enc_map,
        "global_mean": float(global_mean),
        "global_prior": global_prior,
        "broker_multi": broker_multi,
        "region_multi": region_multi,
        "model_version": "retrained",
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    print(f"[retrain] Model saved to {out}  ({out.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain insurance bundle model")
    parser.add_argument("--train", default=os.environ.get("TRAIN_CSV_PATH", "data/train.csv"),
                        help="Path to training CSV (default: data/train.csv or $TRAIN_CSV_PATH)")
    parser.add_argument("--out", default="backend/model.pkl",
                        help="Output path for model.pkl (default: backend/model.pkl)")
    args = parser.parse_args()
    train(args.train, args.out)


if __name__ == "__main__":
    main()
