# ──────────────────────────────────────────────────────────────────────────
# solution.py — DataQuest Hackathon · Insurance Bundle Recommendation
# Model: LightGBM · 150 trees · class_weight="balanced"
# Predictions: integers 0-9 (Purchased_Coverage_Bundle)
# ──────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

_DIR = os.path.dirname(os.path.abspath(__file__))

_MONTH_MAP = {
    "January": 1, "February": 2, "March": 3,    "April": 4,
    "May": 5,     "June": 6,     "July": 7,      "August": 8,
    "September": 9,"October": 10,"November": 11, "December": 12,
}
_DED_MAP = {
    "Tier_1_High_Ded": 1, "Tier_2_Mid_Ded": 2,
    "Tier_3_Low_Ded": 3,  "Tier_4_Zero_Ded": 4,
}


def preprocess(df):
    """Transform raw test.csv into the feature matrix. Keeps User_ID."""
    df = df.copy()
    user_ids = df["User_ID"].copy() if "User_ID" in df.columns else None

    # Drop useless columns
    drop_cols = ["User_ID", "Payment_Schedule", "Policy_Start_Day",
                 "Purchased_Coverage_Bundle"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Fill missing
    df["Child_Dependents"]    = df["Child_Dependents"].fillna(0)
    df["Deductible_Tier"]     = df["Deductible_Tier"].fillna("Tier_1_High_Ded")
    df["Region_Code"]         = df["Region_Code"].fillna("UNKNOWN")
    df["Acquisition_Channel"] = df["Acquisition_Channel"].fillna("Aggregator_Site")

    # Dependents
    df["Total_Dependents"] = (df["Adult_Dependents"] + df["Child_Dependents"]
                              + df["Infant_Dependents"])
    df["Has_Children"] = ((df["Child_Dependents"] + df["Infant_Dependents"]) > 0).astype(int)

    # Binary flags
    df["Employer_ID_Present"]  = df["Employer_ID"].notna().astype(int)
    df["Broker_ID_Missing"]    = df["Broker_ID"].isna().astype(int)
    df["Underwriting_Delayed"] = (df["Underwriting_Processing_Days"] > 0).astype(int)
    df["Fast_Buyer"]           = (df["Days_Since_Quote"] < 7).astype(int)

    # Income
    df["log_Income"]               = np.log1p(df["Estimated_Annual_Income"])
    df["log_Income_Per_Dependent"] = np.log1p(
        df["Estimated_Annual_Income"] / (df["Total_Dependents"] + 1)
    )

    # Risk & interactions
    df["Risk_Score"]       = (df["Previous_Claims_Filed"] + df["Grace_Period_Extensions"]
                              - df["Years_Without_Claims"])
    df["Indecision_Score"] = df["Days_Since_Quote"] * df["Policy_Amendments_Count"]
    df["Policy_Complexity"]= df["Custom_Riders_Requested"] + df["Policy_Amendments_Count"]

    # Temporal
    df["Month_Num"] = df["Policy_Start_Month"].map(_MONTH_MAP)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    df["Week_Sin"]  = np.sin(2 * np.pi * df["Policy_Start_Week"] / 52)
    df["Week_Cos"]  = np.cos(2 * np.pi * df["Policy_Start_Week"] / 52)

    # Ordinal / binary encoding
    df["Deductible_Tier_Ord"] = df["Deductible_Tier"].map(_DED_MAP).fillna(1).astype(int)
    df["Agency_National"]     = (df["Broker_Agency_Type"] == "National_Corporate").astype(int)

    # Employment dummies (baseline = Contractor)
    df["Emp_Employed_FullTime"] = (df["Employment_Status"] == "Employed_FullTime").astype(int)
    df["Emp_Self_Employed"]     = (df["Employment_Status"] == "Self_Employed").astype(int)
    df["Emp_Unemployed"]        = (df["Employment_Status"] == "Unemployed").astype(int)

    # Acquisition dummies (baseline = Affiliate_Group)
    df["Acq_Aggregator_Site"]   = (df["Acquisition_Channel"] == "Aggregator_Site").astype(int)
    df["Acq_Corporate_Partner"] = (df["Acquisition_Channel"] == "Corporate_Partner").astype(int)
    df["Acq_Direct_Website"]    = (df["Acquisition_Channel"] == "Direct_Website").astype(int)
    df["Acq_Local_Broker"]      = (df["Acquisition_Channel"] == "Local_Broker").astype(int)

    # Drop raw columns
    drop_raw = ["Policy_Start_Month", "Employment_Status", "Acquisition_Channel",
                "Deductible_Tier", "Broker_Agency_Type", "Estimated_Annual_Income"]
    df = df.drop(columns=[c for c in drop_raw if c in df.columns])

    # Re-attach User_ID (kept for predict)
    if user_ids is not None:
        df.insert(0, "User_ID", user_ids.values)

    return df


def load_model():
    """Load payload from model.pkl and reconstruct LightGBM Booster."""
    payload = joblib.load(os.path.join(_DIR, "model.pkl"))
    payload["booster"] = lgb.Booster(model_str=payload["model_str"])
    return payload


def predict(df, model):
    """
    Apply target encoding, align to feature_cols, predict with thresholds.
    Returns DataFrame: User_ID | Purchased_Coverage_Bundle (int 0-9).
    """
    booster      = model["booster"]
    thresholds   = model["thresholds"]
    feature_cols = model["feature_cols"]
    global_mean  = model["global_mean"]

    user_ids = df["User_ID"].values
    df = df.copy()

    # Target encoding
    df["Broker_ID_TargetEnc"]   = (df["Broker_ID"].fillna(-1)
                                   .map(model["broker_enc_map"])
                                   .fillna(global_mean))
    df["Region_TargetEnc"]      = (df["Region_Code"]
                                   .map(model["region_enc_map"])
                                   .fillna(global_mean))
    df["Employer_ID_TargetEnc"] = (df["Employer_ID"].fillna(-1)
                                   .map(model["employer_enc_map"])
                                   .fillna(global_mean))

    # Align features and predict
    X     = df[feature_cols].values
    proba = booster.predict(X)
    preds = (proba / thresholds).argmax(axis=1)

    return pd.DataFrame({
        "User_ID": user_ids,
        "Purchased_Coverage_Bundle": preds.astype(int),
    })
