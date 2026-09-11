"""Preprocessing factories used by classical models."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def scaled_pipeline(estimator) -> Pipeline:
    """Create a leakage-safe scaling and estimation pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

