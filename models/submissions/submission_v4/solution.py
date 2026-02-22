"""
solution.py — DataQuest submission v2
- 75-tree LightGBM (optimised for F1 × latency_penalty)
- Raw argmax predictions (no overfitted threshold tuning)
- Full preprocessing from raw CSV (no embedded data files)
- Returns integers 0–9 as required
"""

import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

_DIR = os.path.dirname(os.path.abspath(__file__))

MONTH_ORDER = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
}
DEDUCTIBLE_MAP = {
    'Tier_1_High_Ded': 1, 'Tier_2_Mid_Ded': 2,
    'Tier_3_Low_Ded': 3, 'Tier_4_Zero_Ded': 4,
}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw test CSV into feature matrix (User_ID retained)."""
    # --- Keep User_ID for output ---
    user_ids = df['User_ID'].copy() if 'User_ID' in df.columns else None

    # --- Drop unused columns ---
    drop_cols = ['User_ID', 'Payment_Schedule', 'Policy_Start_Day',
                 'Purchased_Coverage_Bundle']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --- Fill missing values ---
    df['Child_Dependents'] = df['Child_Dependents'].fillna(0)
    df['Deductible_Tier'] = df['Deductible_Tier'].fillna('Tier_1_High_Ded')
    df['Acquisition_Channel'] = df['Acquisition_Channel'].fillna('Aggregator_Site')
    df['Region_Code'] = df['Region_Code'].fillna('UNKNOWN')
    # Broker_ID and Employer_ID — keep NaN for encoding step

    # --- Numeric fill ---
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(0)

    # --- Engineered features ---
    df['Total_Dependents'] = (df['Adult_Dependents']
                              + df['Child_Dependents'].fillna(0)
                              + df['Infant_Dependents'])
    df['Has_Children'] = ((df['Child_Dependents'].fillna(0)
                           + df['Infant_Dependents']) > 0).astype(int)
    df['Employer_ID_Present'] = df['Employer_ID'].notna().astype(int) \
        if 'Employer_ID' in df.columns else 0
    df['Broker_ID_Missing'] = df['Broker_ID'].isna().astype(int) \
        if 'Broker_ID' in df.columns else 0
    df['Underwriting_Delayed'] = (df['Underwriting_Processing_Days'] > 0).astype(int)
    df['Fast_Buyer'] = (df['Days_Since_Quote'] < 7).astype(int)
    income = df['Estimated_Annual_Income'].clip(lower=0)
    df['log_Income'] = np.log1p(income)
    df['log_Income_Per_Dependent'] = np.log1p(income / (df['Total_Dependents'] + 1))
    df['Risk_Score'] = (df['Previous_Claims_Filed']
                        - df['Years_Without_Claims']
                        + df['Grace_Period_Extensions'])
    df['Indecision_Score'] = df['Days_Since_Quote'] * df['Policy_Amendments_Count']
    df['Policy_Complexity'] = df['Custom_Riders_Requested'] + df['Policy_Amendments_Count']

    # --- Month cyclical encoding ---
    df['Month_Num'] = df['Policy_Start_Month'].map(MONTH_ORDER).fillna(1).astype(int)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)

    # --- Week cyclical encoding ---
    df['Week_Sin'] = np.sin(2 * np.pi * df['Policy_Start_Week'] / 52)
    df['Week_Cos'] = np.cos(2 * np.pi * df['Policy_Start_Week'] / 52)

    # --- Ordinal encoding ---
    df['Deductible_Tier_Ord'] = df['Deductible_Tier'].map(DEDUCTIBLE_MAP).fillna(1).astype(int)

    # --- Binary encoding ---
    df['Agency_National'] = (df['Broker_Agency_Type'] == 'National_Corporate').astype(int) \
        if 'Broker_Agency_Type' in df.columns else 0

    # --- Employment one-hot ---
    emp_col = df['Employment_Status'] if 'Employment_Status' in df.columns \
        else pd.Series('Employed_FullTime', index=df.index)
    df['Emp_Employed_FullTime'] = (emp_col == 'Employed_FullTime').astype(int)
    df['Emp_Self_Employed']     = (emp_col == 'Self_Employed').astype(int)
    df['Emp_Unemployed']        = (emp_col == 'Unemployed').astype(int)

    # --- Acquisition channel one-hot ---
    acq_col = df['Acquisition_Channel'] if 'Acquisition_Channel' in df.columns \
        else pd.Series('Aggregator_Site', index=df.index)
    df['Acq_Aggregator_Site']  = (acq_col == 'Aggregator_Site').astype(int)
    df['Acq_Corporate_Partner'] = (acq_col == 'Corporate_Partner').astype(int)
    df['Acq_Direct_Website']   = (acq_col == 'Direct_Website').astype(int)
    df['Acq_Local_Broker']     = (acq_col == 'Local_Broker').astype(int)

    # --- Re-attach User_ID (needed by predict()) ---
    if user_ids is not None:
        df['User_ID'] = user_ids.values

    return df


def load_model():
    """Load model.pkl and reconstruct LightGBM booster."""
    payload = joblib.load(os.path.join(_DIR, 'model.pkl'))
    payload['booster'] = lgb.Booster(model_str=payload['model_str'])
    return payload


def predict(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Apply target encoding, run inference, return integer predictions."""
    booster          = model['booster']
    feature_cols     = model['feature_cols']
    broker_enc_map   = model['broker_enc_map']
    region_enc_map   = model['region_enc_map']
    employer_enc_map = model['employer_enc_map']
    global_mean      = model['global_mean']

    # Extract User_ID before encoding
    user_ids = df['User_ID'].values if 'User_ID' in df.columns else None

    # --- Target encoding ---
    if 'Broker_ID' in df.columns:
        df['Broker_ID_TargetEnc'] = (df['Broker_ID']
                                     .map(broker_enc_map)
                                     .fillna(global_mean))
    else:
        df['Broker_ID_TargetEnc'] = global_mean

    if 'Region_Code' in df.columns:
        df['Region_TargetEnc'] = (df['Region_Code']
                                  .map(region_enc_map)
                                  .fillna(global_mean))
    else:
        df['Region_TargetEnc'] = global_mean

    if 'Employer_ID' in df.columns:
        df['Employer_ID_TargetEnc'] = (df['Employer_ID']
                                       .map(employer_enc_map)
                                       .fillna(global_mean))
    else:
        df['Employer_ID_TargetEnc'] = global_mean

    # --- Build feature matrix ---
    X = df.reindex(columns=feature_cols, fill_value=0).values

    # --- Predict (raw argmax — no threshold tuning) ---
    proba = booster.predict(X)
    preds = proba.argmax(axis=1).astype(int)

    return pd.DataFrame({
        'User_ID': user_ids if user_ids is not None else range(len(preds)),
        'Purchased_Coverage_Bundle': preds,
    })
