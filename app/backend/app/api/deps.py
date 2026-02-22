"""FastAPI dependencies."""
from fastapi import HTTPException

from app.core.model_loader import ModelStore
from app.core.model_loader import LoadedModel


def get_model() -> LoadedModel:
    """Require model to be loaded. Raise 503 if not."""
    model = ModelStore.get()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model
