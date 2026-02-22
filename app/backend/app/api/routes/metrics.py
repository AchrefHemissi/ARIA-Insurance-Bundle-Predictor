"""Runtime metrics endpoint."""
from fastapi import APIRouter

from app.core.metrics import MetricsStore

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def metrics():
    """Runtime stats: total predictions, rolling avg latency, bundle distribution, error rate."""
    return MetricsStore.get()
