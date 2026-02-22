"""Application configuration."""
import os
from pathlib import Path

# Model
MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parent.parent / "model.pkl"))

# Prediction
N_CLASSES = 10
CONF_HIGH = 0.6
CONF_MEDIUM = 0.35

# Metrics
METRICS_ROLLING_WINDOW = 1000
