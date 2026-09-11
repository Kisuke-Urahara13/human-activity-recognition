"""Shared model evaluation and report persistence."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data_loader import ACTIVITY_NAMES


def evaluate_predictions(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def save_evaluation(y_true, y_pred, model_name: str, report_dir: str | Path) -> dict:
    report_dir = Path(report_dir)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_predictions(y_true, y_pred)
    labels = sorted(ACTIVITY_NAMES)
    names = [ACTIVITY_NAMES[label] for label in labels]
    details = classification_report(
        y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0
    )
    with (report_dir / f"{model_name}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"summary": metrics, "classification_report": details}, file, indent=2)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / f"{model_name}_confusion_matrix.png", dpi=160)
    plt.close()
    return metrics

