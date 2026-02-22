"""SHAP-based explainability."""
import numpy as np


def top_shap_features(
    shap_values: list | np.ndarray,
    predicted_class: int,
    feature_cols: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Extract top-k features by absolute SHAP value for the predicted class."""
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[predicted_class]).reshape(-1)
    else:
        sv = np.asarray(shap_values[0]).reshape(-1)
    n_feat = min(len(sv), len(feature_cols))
    contribs = [(feature_cols[i], float(np.asarray(sv[i]).item())) for i in range(n_feat)]
    contribs.sort(key=lambda x: -abs(x[1]))
    return [
        {"feature": c, "shap_value": round(v, 6), "direction": "positive" if v > 0 else "negative"}
        for c, v in contribs[:top_k]
    ]
