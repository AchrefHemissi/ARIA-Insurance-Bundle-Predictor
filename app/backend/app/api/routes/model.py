"""Model metadata and feature importances."""
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_model
from app.config import N_CLASSES
from app.core.model_loader import LoadedModel, ModelStore

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info")
def model_info(model: LoadedModel = Depends(get_model)):
    """Static metadata: version, features, hyperparams, encoding strategy."""
    params = getattr(model.booster, "params", {}) or {}
    return {
        "model_version": model.model_version,
        "num_features": model.num_features,
        "num_classes": N_CLASSES,
        "feature_columns": model.feature_cols,
        "hyperparameters": {
            "num_leaves": params.get("num_leaves", 95),
            "learning_rate": params.get("learning_rate", 0.07),
        },
        "encoding_strategy": (
            "Multi-class target encoding for Broker_ID and Region_Code (10 features each), "
            "smoothed mean-target for Employer_ID"
        ),
    }


@router.get("/features")
def model_features():
    """Feature importances (gain), full dict and top-10."""
    imp = ModelStore.get_feature_importances()
    if not imp:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return imp
