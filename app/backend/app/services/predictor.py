"""Inference: encoding + model prediction."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.config import CONF_HIGH, CONF_MEDIUM, N_CLASSES


@dataclass
class PredictionResult:
    """Single prediction output."""

    predicted_bundle: int
    probabilities: list[float]
    confidence: float
    confidence_label: str
    inference_latency_ms: float


def confidence_label(conf: float) -> str:
    if conf >= CONF_HIGH:
        return "High"
    if conf >= CONF_MEDIUM:
        return "Medium"
    return "Low"


def encode_and_predict(
    df: pd.DataFrame,
    booster: Any,
    feature_cols: list[str],
    employer_enc_map: dict,
    global_mean: float,
    global_prior: list[float],
    broker_multi: dict,
    region_multi: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply target encoding and run inference. Returns (proba, X)."""
    df = df.copy()
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

    X = df.reindex(columns=feature_cols, fill_value=0).values
    proba = booster.predict(X)
    return proba, X
