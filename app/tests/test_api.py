"""Integration tests for the FastAPI application endpoints.

These tests use FastAPI's TestClient (backed by httpx) so no running server
is needed.  The real model is NOT loaded — ModelStore is patched to return
the lightweight mock.
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient

from tests.conftest import SAMPLE_ROW


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
N_CLASSES = 10
N_FEATURES = 40


def _make_mock_loaded_model():
    """Build a mock LoadedModel for patching ModelStore."""
    model = MagicMock()
    model.feature_cols = [f"feat_{i}" for i in range(N_FEATURES)]
    model.num_features = N_FEATURES
    model.employer_enc_map = {}
    model.global_mean = 0.5
    model.global_prior = [1.0 / N_CLASSES] * N_CLASSES
    model.broker_multi = {cls: {} for cls in range(N_CLASSES)}
    model.region_multi = {cls: {} for cls in range(N_CLASSES)}
    model.model_version = "test"

    def _predict(X):
        proba = np.full((X.shape[0], N_CLASSES), 1.0 / N_CLASSES)
        return proba

    model.booster = MagicMock()
    model.booster.predict = _predict

    # SHAP explainer mock
    model.shap_explainer = MagicMock()
    model.shap_explainer.shap_values.return_value = [
        np.random.randn(1, N_FEATURES) for _ in range(N_CLASSES)
    ]
    return model


@pytest.fixture
def client():
    """Create a TestClient with a mocked model."""
    mock_model = _make_mock_loaded_model()

    with patch("app.core.model_loader.ModelStore") as MockStore:
        MockStore.get.return_value = mock_model
        MockStore.load.return_value = True
        MockStore.uptime_seconds.return_value = 42.0
        MockStore.get_feature_importances.return_value = {
            "full": {}, "top_10": [],
        }

        # Also patch the dependency that routes use
        with patch("app.api.deps.get_model", return_value=mock_model):
            from app.main import app
            yield TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_fields(self, client):
        data = client.get("/health").json()
        assert "model_loaded" in data
        assert "num_features" in data
        assert "num_classes" in data
        assert "uptime_seconds" in data


class TestPredictEndpoint:
    def test_predict_valid_payload(self, client):
        resp = client.post("/predict", json=SAMPLE_ROW)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_bundle" in data
        assert "probabilities" in data
        assert "confidence" in data
        assert "confidence_label" in data
        assert "inference_latency_ms" in data

    def test_predict_returns_valid_bundle(self, client):
        data = client.post("/predict", json=SAMPLE_ROW).json()
        assert 0 <= data["predicted_bundle"] < N_CLASSES

    def test_predict_probabilities_length(self, client):
        data = client.post("/predict", json=SAMPLE_ROW).json()
        assert len(data["probabilities"]) == N_CLASSES

    def test_predict_empty_payload_fails(self, client):
        resp = client.post("/predict", json={})
        # The preprocessor will fail on missing columns → 400
        assert resp.status_code == 400


class TestPredictExplainEndpoint:
    def test_explain_returns_shap(self, client):
        resp = client.post("/predict/explain", json=SAMPLE_ROW)
        assert resp.status_code == 200
        data = resp.json()
        assert "top_5_shap_features" in data
        assert len(data["top_5_shap_features"]) <= 5
