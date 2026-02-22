"""Runtime metrics storage."""
from app.config import METRICS_ROLLING_WINDOW, N_CLASSES


class MetricsStore:
    """In-memory store for prediction metrics."""

    _total_predictions: int = 0
    _prediction_errors: int = 0
    _latencies_ms: list[float] = []
    _bundle_counts: dict[int, int] = {k: 0 for k in range(N_CLASSES)}

    @classmethod
    def record_prediction(cls, latency_ms: float, pred_bundle: int) -> None:
        cls._total_predictions += 1
        cls._latencies_ms.append(latency_ms)
        if len(cls._latencies_ms) > METRICS_ROLLING_WINDOW:
            cls._latencies_ms.pop(0)
        cls._bundle_counts[pred_bundle] = cls._bundle_counts.get(pred_bundle, 0) + 1

    @classmethod
    def record_error(cls) -> None:
        cls._prediction_errors += 1

    @classmethod
    def record_batch(cls, total_latency_ms: float, preds: list[int]) -> None:
        cls._total_predictions += len(preds)
        cls._latencies_ms.append(total_latency_ms)
        if len(cls._latencies_ms) > METRICS_ROLLING_WINDOW:
            cls._latencies_ms.pop(0)
        for p in preds:
            cls._bundle_counts[p] = cls._bundle_counts.get(p, 0) + 1

    @classmethod
    def get(cls) -> dict:
        n = len(cls._latencies_ms)
        avg_latency = sum(cls._latencies_ms) / n if n > 0 else 0.0
        total = cls._total_predictions + cls._prediction_errors
        error_rate = cls._prediction_errors / total if total > 0 else 0.0
        return {
            "total_predictions": cls._total_predictions,
            "rolling_avg_latency_ms": round(avg_latency, 2),
            "bundle_distribution": cls._bundle_counts.copy(),
            "error_rate": round(error_rate, 4),
        }
