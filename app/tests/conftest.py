"""Shared pytest fixtures for tests."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Add backend to sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


# ---------------------------------------------------------------------------
# Sample input data (mimics a single client row from the frontend)
# ---------------------------------------------------------------------------
SAMPLE_ROW = {
    "User_ID": 12345,
    "Age": 35,
    "Annual_Income": 55000,
    "Employment_Status": "Employed",
    "Adult_Dependents": 1,
    "Child_Dependents": 2,
    "Infant_Dependents": 0,
    "Previous_Claims_Filed": 1,
    "Years_Without_Claims": 3,
    "Grace_Period_Extensions": 0,
    "Days_Since_Quote": 5,
    "Custom_Riders": 2,
    "Policy_Amendments_Count": 0,
    "Policy_Start_Month": 3,
    "Policy_Start_Week": 10,
    "Deductible_Tier": "Medium",
    "Acquisition_Channel": "Online",
    "Existing_Policyholder": 0,
    "Num_Vehicles": 1,
}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Single-row DataFrame matching frontend payload structure."""
    return pd.DataFrame([SAMPLE_ROW])


@pytest.fixture
def sample_df_batch() -> pd.DataFrame:
    """3-row DataFrame for batch testing."""
    rows = []
    for i in range(3):
        row = SAMPLE_ROW.copy()
        row["User_ID"] = 10000 + i
        row["Age"] = 30 + i * 5
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mock model (avoids loading the real 2 MB model.pkl in unit tests)
# ---------------------------------------------------------------------------
N_CLASSES = 10
N_FEATURES = 40  # approximate number of features the model expects


@pytest.fixture
def mock_model():
    """Lightweight mock that mimics LoadedModel enough for unit tests."""
    model = MagicMock()
    model.feature_cols = [f"feat_{i}" for i in range(N_FEATURES)]
    model.num_features = N_FEATURES
    model.employer_enc_map = {}
    model.global_mean = 0.5
    model.global_prior = [1.0 / N_CLASSES] * N_CLASSES
    model.broker_multi = {cls: {} for cls in range(N_CLASSES)}
    model.region_multi = {cls: {} for cls in range(N_CLASSES)}

    # booster.predict returns uniform probabilities
    def _predict(X):
        return np.full((X.shape[0], N_CLASSES), 1.0 / N_CLASSES)

    model.booster = MagicMock()
    model.booster.predict = _predict
    return model
