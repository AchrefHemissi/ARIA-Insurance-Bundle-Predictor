"""Prediction endpoints."""
import time
from io import BytesIO

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_model
from app.config import N_CLASSES
from app.core.metrics import MetricsStore
from app.core.model_loader import LoadedModel
from app.schemas.schemas import PredictRequest
from app.services.explainer import top_shap_features
from app.services.preprocessor import preprocess
from app.services.predictor import confidence_label, encode_and_predict

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("")
def predict(req: PredictRequest, model: LoadedModel = Depends(get_model)):
    """Single prediction from raw client JSON."""
    t0 = time.perf_counter()
    try:
        df = pd.DataFrame([req.model_dump()])
        df = preprocess(df)
        proba, _ = encode_and_predict(
            df, model.booster,
            model.feature_cols,
            model.employer_enc_map,
            model.global_mean,
            model.global_prior,
            model.broker_multi,
            model.region_multi,
        )
        pred_bundle = int(proba[0].argmax())
        conf = float(proba[0].max())
        latency_ms = (time.perf_counter() - t0) * 1000
        MetricsStore.record_prediction(latency_ms, pred_bundle)
        return {
            "predicted_bundle": pred_bundle,
            "probabilities": [round(float(p), 6) for p in proba[0]],
            "confidence": round(conf, 4),
            "confidence_label": confidence_label(conf),
            "inference_latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        MetricsStore.record_error()
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/explain")
def predict_explain(req: PredictRequest, model: LoadedModel = Depends(get_model)):
    """Same as /predict plus top 5 SHAP features."""
    t0 = time.perf_counter()
    try:
        df = pd.DataFrame([req.model_dump()])
        df = preprocess(df)
        proba, X = encode_and_predict(
            df, model.booster,
            model.feature_cols,
            model.employer_enc_map,
            model.global_mean,
            model.global_prior,
            model.broker_multi,
            model.region_multi,
        )
        pred_bundle = int(proba[0].argmax())
        conf = float(proba[0].max())
        shap_values = model.shap_explainer.shap_values(X)
        top5 = top_shap_features(shap_values, pred_bundle, model.feature_cols, top_k=5)
        latency_ms = (time.perf_counter() - t0) * 1000
        MetricsStore.record_prediction(latency_ms, pred_bundle)
        return {
            "predicted_bundle": pred_bundle,
            "probabilities": [round(float(p), 6) for p in proba[0]],
            "confidence": round(conf, 4),
            "confidence_label": confidence_label(conf),
            "inference_latency_ms": round(latency_ms, 2),
            "top_5_shap_features": top5,
        }
    except Exception as e:
        MetricsStore.record_error()
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/batch")
async def predict_batch(file: UploadFile = File(...), model: LoadedModel = Depends(get_model)):
    """Batch prediction from CSV upload."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    t0 = time.perf_counter()
    try:
        contents = await file.read()
        df_raw = pd.read_csv(BytesIO(contents))
        user_ids = df_raw["User_ID"].values if "User_ID" in df_raw.columns else list(range(len(df_raw)))
        df = preprocess(df_raw)
        proba, _ = encode_and_predict(
            df, model.booster,
            model.feature_cols,
            model.employer_enc_map,
            model.global_mean,
            model.global_prior,
            model.broker_multi,
            model.region_multi,
        )
        preds = proba.argmax(axis=1).astype(int)
        confs = proba.max(axis=1)
        results = [
            {"User_ID": int(uid), "predicted_bundle": int(p), "confidence": round(float(c), 4)}
            for uid, p, c in zip(user_ids, preds, confs)
        ]
        batch_dist = {str(k): int((preds == k).sum()) for k in range(N_CLASSES)}
        total_latency_ms = (time.perf_counter() - t0) * 1000
        MetricsStore.record_batch(total_latency_ms, preds.tolist())
        return {
            "rows_processed": len(preds),
            "results": results,
            "bundle_distribution": batch_dist,
            "average_confidence": round(float(confs.mean()), 4),
            "total_latency_ms": round(total_latency_ms, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
