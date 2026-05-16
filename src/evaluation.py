"""Model evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(model: Any, x_test: Any, y_test: Any) -> dict[str, Any]:
    """Return standard classification metrics for a fitted model."""
    predictions = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Save metrics as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")


def save_confusion_matrix(model: Any, x_test: Any, y_test: Any, output_path: str | Path) -> None:
    """Save a confusion matrix plot for the fitted model."""
    predictions = model.predict(x_test)
    labels = sorted(set(y_test) | set(predictions))
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Wafer Defect Classification Confusion Matrix")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar values to JSON-friendly Python values."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
