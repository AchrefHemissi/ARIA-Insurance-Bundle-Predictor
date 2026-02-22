"""Health check endpoint."""
from fastapi import APIRouter

from app.config import N_CLASSES
from app.core.model_loader import ModelStore

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Returns model status, feature count, class count, uptime."""
    model = ModelStore.get()
    return {
        "model_loaded": model is not None,
        "num_features": model.num_features if model else 0,
        "num_classes": N_CLASSES,
        "uptime_seconds": ModelStore.uptime_seconds(),
    }
