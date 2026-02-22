"""Request and response schemas."""
from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Accepts any JSON with raw client fields."""

    model_config = {"extra": "allow"}


class HealthResponse(BaseModel):
    model_loaded: bool
    num_features: int
    num_classes: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_predictions: int
    rolling_avg_latency_ms: float
    bundle_distribution: dict[str, int]
    error_rate: float


class PredictResponse(BaseModel):
    predicted_bundle: int
    probabilities: list[float]
    confidence: float
    confidence_label: str
    inference_latency_ms: float


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictExplainResponse(PredictResponse):
    top_5_shap_features: list[ShapFeature]


class BatchResultItem(BaseModel):
    User_ID: int
    predicted_bundle: int
    confidence: float


class PredictBatchResponse(BaseModel):
    rows_processed: int
    results: list[BatchResultItem]
    bundle_distribution: dict[str, int]
    average_confidence: float
    total_latency_ms: float
