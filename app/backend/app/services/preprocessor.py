"""Feature engineering — mirrors solution.py preprocessing exactly."""
import pandas as pd
import numpy as np

from app.config import N_CLASSES

MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}
DEDUCTIBLE_MAP = {
    "Tier_1_High_Ded": 1, "Tier_2_Mid_Ded": 2, "Tier_3_Low_Ded": 3, "Tier_4_Zero_Ded": 4,
}

# ── Frontend → Internal translation maps ────────────────────────────────────

# Deductible_Tier: frontend sends "Low"/"Medium"/"High"
_DEDUCTIBLE_FRONTEND_MAP = {
    "High":   "Tier_1_High_Ded",
    "Medium": "Tier_2_Mid_Ded",
    "Low":    "Tier_3_Low_Ded",
}

# Acquisition_Channel: frontend sends "Online"/"Broker"/"Referral"/"Corporate_Partner"
_ACQ_FRONTEND_MAP = {
    "Online":            "Aggregator_Site",
    "Broker":            "Local_Broker",
    "Referral":          "Aggregator_Site",
    "Corporate_Partner": "Corporate_Partner",
}

# Employment_Status: frontend sends 1 (employed) or 0 (unemployed)
_EMPLOYMENT_FRONTEND_MAP = {
    1:                  "Employed_FullTime",
    0:                  "Unemployed",
    "Employed":         "Employed_FullTime",
    "Unemployed":       "Unemployed",
    "Employed_FullTime":"Employed_FullTime",
    "Self_Employed":    "Self_Employed",
}

# Month: frontend sends int 1-12; preprocessor expects month-name string
_MONTH_INT_MAP = {i: name for name, i in MONTH_ORDER.items()}


def _normalize_frontend_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Translate frontend field names / value vocabularies to internal training column names."""
    df = df.copy()

    # Annual_Income → Estimated_Annual_Income
    if "Annual_Income" in df.columns and "Estimated_Annual_Income" not in df.columns:
        df["Estimated_Annual_Income"] = df["Annual_Income"]

    # Custom_Riders → Custom_Riders_Requested
    if "Custom_Riders" in df.columns and "Custom_Riders_Requested" not in df.columns:
        df["Custom_Riders_Requested"] = df["Custom_Riders"]

    # Vehicles → Num_Vehicles (kept for completeness, not used by model features)
    if "Vehicles" in df.columns and "Num_Vehicles" not in df.columns:
        df["Num_Vehicles"] = df["Vehicles"]

    # Employment_Status: int or plain string → training string
    if "Employment_Status" in df.columns:
        df["Employment_Status"] = df["Employment_Status"].map(
            lambda v: _EMPLOYMENT_FRONTEND_MAP.get(v, "Employed_FullTime")
        )

    # Policy_Start_Month: int 1-12 → month-name string
    if "Policy_Start_Month" in df.columns:
        df["Policy_Start_Month"] = df["Policy_Start_Month"].map(
            lambda v: _MONTH_INT_MAP.get(v, v) if isinstance(v, int) else v
        )

    # Deductible_Tier: "Low"/"Medium"/"High" → "Tier_N_..." strings
    if "Deductible_Tier" in df.columns:
        df["Deductible_Tier"] = df["Deductible_Tier"].map(
            lambda v: _DEDUCTIBLE_FRONTEND_MAP.get(v, v)
        )

    # Acquisition_Channel: frontend labels → training labels
    if "Acquisition_Channel" in df.columns:
        df["Acquisition_Channel"] = df["Acquisition_Channel"].map(
            lambda v: _ACQ_FRONTEND_MAP.get(v, v)
        )

    # Ensure all columns referenced by preprocess() exist with safe defaults.
    # The frontend only sends ~16 fields; the model was trained on CSV data
    # with many more columns. Missing ones get sensible defaults here.
    for col, default in [
        ("Region_Code", "UNKNOWN"),
        ("Underwriting_Processing_Days", 0),
        ("Policy_Start_Week", 1),
        ("Policy_Amendments_Count", 0),
        ("Existing_Policyholder", 0),
        ("Num_Vehicles", 1),
    ]:
        if col not in df.columns:
            df[col] = default

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering — exact mirror of solution.py preprocess()."""
    # Translate frontend-friendly field names/values first
    df = _normalize_frontend_fields(df)

    user_ids = df["User_ID"].copy() if "User_ID" in df.columns else None

    drop_cols = ["User_ID", "Payment_Schedule", "Policy_Start_Day", "Purchased_Coverage_Bundle"]
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
    df["Agency_National"] = (df["Broker_Agency_Type"] == "National_Corporate").astype(int) if "Broker_Agency_Type" in df.columns else 0

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
    return df
