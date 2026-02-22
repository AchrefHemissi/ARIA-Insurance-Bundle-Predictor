"""Unit tests for backend/app/services/predictor.py."""
import numpy as np
import pandas as pd
import pytest

from app.services.predictor import confidence_label, encode_and_predict


class TestConfidenceLabel:
    """Verify confidence_label thresholds (High ≥ 0.6, Medium ≥ 0.35, else Low)."""

    def test_high_confidence(self):
        assert confidence_label(0.85) == "High"
        assert confidence_label(0.60) == "High"

    def test_medium_confidence(self):
        assert confidence_label(0.50) == "Medium"
        assert confidence_label(0.35) == "Medium"

    def test_low_confidence(self):
        assert confidence_label(0.20) == "Low"
        assert confidence_label(0.0) == "Low"


class TestEncodeAndPredict:
    """Verify encode_and_predict returns correct shapes."""

    def test_output_shape_single(self, sample_df, mock_model):
        from app.services.preprocessor import preprocess

        df = preprocess(sample_df)
        proba, X = encode_and_predict(
            df,
            mock_model.booster,
            mock_model.feature_cols,
            mock_model.employer_enc_map,
            mock_model.global_mean,
            mock_model.global_prior,
            mock_model.broker_multi,
            mock_model.region_multi,
        )
        assert proba.shape == (1, 10), f"Expected (1, 10) but got {proba.shape}"
        assert X.shape[0] == 1

    def test_output_shape_batch(self, sample_df_batch, mock_model):
        from app.services.preprocessor import preprocess

        df = preprocess(sample_df_batch)
        proba, X = encode_and_predict(
            df,
            mock_model.booster,
            mock_model.feature_cols,
            mock_model.employer_enc_map,
            mock_model.global_mean,
            mock_model.global_prior,
            mock_model.broker_multi,
            mock_model.region_multi,
        )
        assert proba.shape == (3, 10)
        assert X.shape[0] == 3

    def test_probabilities_sum_to_one(self, sample_df, mock_model):
        from app.services.preprocessor import preprocess

        df = preprocess(sample_df)
        proba, _ = encode_and_predict(
            df,
            mock_model.booster,
            mock_model.feature_cols,
            mock_model.employer_enc_map,
            mock_model.global_mean,
            mock_model.global_prior,
            mock_model.broker_multi,
            mock_model.region_multi,
        )
        assert np.isclose(proba[0].sum(), 1.0, atol=1e-4)
