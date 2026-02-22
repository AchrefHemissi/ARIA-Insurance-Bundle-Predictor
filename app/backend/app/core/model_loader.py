"""Model loading and in-memory state."""
import time
from dataclasses import dataclass
from typing import Any

import joblib
import lightgbm as lgb
import shap

from app.config import MODEL_PATH, N_CLASSES


@dataclass
class LoadedModel:
    """Loaded model artifact with booster and encoding maps."""

    booster: lgb.Booster
    feature_cols: list[str]
    employer_enc_map: dict
    global_mean: float
    global_prior: list[float]
    broker_multi: dict
    region_multi: dict
    model_version: str
    shap_explainer: shap.TreeExplainer

    @property
    def num_features(self) -> int:
        return len(self.feature_cols)


class ModelStore:
    """Singleton store for the loaded model and cached artifacts."""

    _instance: "LoadedModel | None" = None
    _feature_importances: dict[str, Any] | None = None
    _start_time: float = 0.0

    @classmethod
    def load(cls) -> bool:
        """Load model from disk. Returns True if successful."""
        cls._start_time = time.perf_counter()
        if not MODEL_PATH.is_file():
            return False
        payload = joblib.load(MODEL_PATH)
        booster = lgb.Booster(model_str=payload["model_str"])
        cls._feature_importances = cls._compute_feature_importances(booster, payload["feature_cols"])
        cls._instance = LoadedModel(
            booster=booster,
            feature_cols=payload["feature_cols"],
            employer_enc_map=payload["employer_enc_map"],
            global_mean=payload["global_mean"],
            global_prior=payload["global_prior"],
            broker_multi=payload["broker_multi"],
            region_multi=payload["region_multi"],
            model_version=payload.get("model_version", "v7"),
            shap_explainer=shap.TreeExplainer(booster),
        )
        return True

    @classmethod
    def _compute_feature_importances(cls, booster: lgb.Booster, cols: list[str]) -> dict[str, Any]:
        imp = booster.feature_importance(importance_type="gain")
        imp_dict = {c: float(v) for c, v in zip(cols, imp)}
        imp_sorted = sorted(imp_dict.items(), key=lambda x: -x[1])
        return {
            "full": imp_dict,
            "top_10": [{"feature": k, "importance": v} for k, v in imp_sorted[:10]],
        }

    @classmethod
    def get(cls) -> "LoadedModel | None":
        return cls._instance

    @classmethod
    def get_feature_importances(cls) -> dict[str, Any] | None:
        return cls._feature_importances

    @classmethod
    def uptime_seconds(cls) -> float:
        return round(time.perf_counter() - cls._start_time, 2) if cls._start_time else 0.0

    @classmethod
    def clear(cls) -> None:
        cls._instance = None
        cls._feature_importances = None
