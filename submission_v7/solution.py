"""solution.py — DataQuest submission v7
Changes vs v6:
- Multi-class target encoding for Broker_ID and Region_Code (10 features each)
- class_weight=None (SMOTE handles imbalance — no double-penalising)
- Tuned hyperparams: num_leaves=95, lr=0.07, reg_lambda=0.5
"""
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

_DIR = os.path.dirname(os.path.abspath(__file__))

MONTH_ORDER = {
    'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
    'July':7,'August':8,'September':9,'October':10,'November':11,'December':12,
}
DEDUCTIBLE_MAP = {
    'Tier_1_High_Ded':1,'Tier_2_Mid_Ded':2,'Tier_3_Low_Ded':3,'Tier_4_Zero_Ded':4,
}
N_CLASSES = 10


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    user_ids = df['User_ID'].copy() if 'User_ID' in df.columns else None

    drop_cols = ['User_ID','Payment_Schedule','Policy_Start_Day','Purchased_Coverage_Bundle']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df['Child_Dependents']    = df['Child_Dependents'].fillna(0)
    df['Deductible_Tier']     = df['Deductible_Tier'].fillna('Tier_1_High_Ded')
    df['Acquisition_Channel'] = df['Acquisition_Channel'].fillna('Aggregator_Site')
    df['Region_Code']         = df['Region_Code'].fillna('UNKNOWN')

    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(0)

    df['Total_Dependents']       = df['Adult_Dependents'] + df['Child_Dependents'].fillna(0) + df['Infant_Dependents']
    df['Has_Children']           = ((df['Child_Dependents'].fillna(0) + df['Infant_Dependents']) > 0).astype(int)
    df['Employer_ID_Present']    = df['Employer_ID'].notna().astype(int) if 'Employer_ID' in df.columns else 0
    df['Broker_ID_Missing']      = df['Broker_ID'].isna().astype(int) if 'Broker_ID' in df.columns else 0
    df['Underwriting_Delayed']   = (df['Underwriting_Processing_Days'] > 0).astype(int)
    df['Fast_Buyer']             = (df['Days_Since_Quote'] < 7).astype(int)
    income = df['Estimated_Annual_Income'].clip(lower=0)
    df['log_Income']             = np.log1p(income)
    df['log_Income_Per_Dependent'] = np.log1p(income / (df['Total_Dependents'] + 1))
    df['Risk_Score']             = df['Previous_Claims_Filed'] - df['Years_Without_Claims'] + df['Grace_Period_Extensions']
    df['Indecision_Score']       = df['Days_Since_Quote'] * df['Policy_Amendments_Count']
    df['Policy_Complexity']      = df['Custom_Riders_Requested'] + df['Policy_Amendments_Count']

    df['Month_Num'] = df['Policy_Start_Month'].map(MONTH_ORDER).fillna(1).astype(int)
    df['Month_Sin'] = np.sin(2*np.pi*df['Month_Num']/12)
    df['Month_Cos'] = np.cos(2*np.pi*df['Month_Num']/12)
    df['Week_Sin']  = np.sin(2*np.pi*df['Policy_Start_Week']/52)
    df['Week_Cos']  = np.cos(2*np.pi*df['Policy_Start_Week']/52)

    df['Deductible_Tier_Ord'] = df['Deductible_Tier'].map(DEDUCTIBLE_MAP).fillna(1).astype(int)
    df['Agency_National']     = (df['Broker_Agency_Type'] == 'National_Corporate').astype(int) if 'Broker_Agency_Type' in df.columns else 0

    emp = df['Employment_Status'] if 'Employment_Status' in df.columns else pd.Series('Employed_FullTime', index=df.index)
    df['Emp_Employed_FullTime'] = (emp == 'Employed_FullTime').astype(int)
    df['Emp_Self_Employed']     = (emp == 'Self_Employed').astype(int)
    df['Emp_Unemployed']        = (emp == 'Unemployed').astype(int)

    acq = df['Acquisition_Channel'] if 'Acquisition_Channel' in df.columns else pd.Series('Aggregator_Site', index=df.index)
    df['Acq_Aggregator_Site']   = (acq == 'Aggregator_Site').astype(int)
    df['Acq_Corporate_Partner'] = (acq == 'Corporate_Partner').astype(int)
    df['Acq_Direct_Website']    = (acq == 'Direct_Website').astype(int)
    df['Acq_Local_Broker']      = (acq == 'Local_Broker').astype(int)

    if user_ids is not None:
        df['User_ID'] = user_ids.values
    return df


def load_model():
    payload = joblib.load(os.path.join(_DIR, 'model.pkl'))
    payload['booster'] = lgb.Booster(model_str=payload['model_str'])
    return payload


def predict(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    booster          = model['booster']
    feature_cols     = model['feature_cols']
    employer_enc_map = model['employer_enc_map']
    global_mean      = model['global_mean']
    global_prior     = model['global_prior']
    broker_multi     = model['broker_multi']   # {cls: {broker_id: prob}}
    region_multi     = model['region_multi']   # {cls: {region_code: prob}}

    user_ids = df['User_ID'].values if 'User_ID' in df.columns else None

    # Employer single encoding
    if 'Employer_ID' in df.columns:
        df['Employer_ID_TargetEnc'] = df['Employer_ID'].map(employer_enc_map).fillna(global_mean)
    else:
        df['Employer_ID_TargetEnc'] = global_mean

    # Multi-class encoding for Broker_ID (10 features)
    broker_raw = df['Broker_ID'].fillna(-1) if 'Broker_ID' in df.columns else pd.Series(-1, index=df.index)
    for cls in range(N_CLASSES):
        df[f'Broker_C{cls}'] = broker_raw.map(broker_multi[cls]).fillna(global_prior[cls])

    # Multi-class encoding for Region_Code (10 features)
    region_raw = df['Region_Code'].fillna('UNKNOWN') if 'Region_Code' in df.columns else pd.Series('UNKNOWN', index=df.index)
    for cls in range(N_CLASSES):
        df[f'Region_C{cls}'] = region_raw.map(region_multi[cls]).fillna(global_prior[cls])

    X    = df.reindex(columns=feature_cols, fill_value=0).values
    proba = booster.predict(X)
    preds = proba.argmax(axis=1).astype(int)

    return pd.DataFrame({
        'User_ID': user_ids if user_ids is not None else range(len(preds)),
        'Purchased_Coverage_Bundle': preds,
    })
