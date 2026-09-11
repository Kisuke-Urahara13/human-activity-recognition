import numpy as np
from sklearn.linear_model import LogisticRegression

from src.evaluate import evaluate_predictions
from src.preprocessing import scaled_pipeline


def test_evaluation_for_perfect_predictions():
    metrics = evaluate_predictions(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_scaled_pipeline_can_fit():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1]])
    y = np.array([1, 2, 1, 2])
    model = scaled_pipeline(LogisticRegression()).fit(X, y)
    assert model.predict(X).shape == (4,)

