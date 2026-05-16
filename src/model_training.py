"""Train a baseline wafer defect classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor, load_wafer_data, split_features_target
from src.evaluation import evaluate_classifier, save_confusion_matrix, save_metrics


def train_model(
    data_path: str | Path = "data/sample_data.csv",
    output_dir: str | Path = "results",
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, float]:
    """Train, evaluate, and save a baseline defect classifier."""
    data = load_wafer_data(data_path)
    features, target = split_features_target(data)

    stratify = target if target.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_classifier(model, x_test, y_test)
    save_metrics(metrics, output_path / "metrics.json")
    save_confusion_matrix(model, x_test, y_test, output_path / "confusion_matrix.png")
    joblib.dump(model, output_path / "defect_classifier.joblib")

    return {
        "accuracy": float(metrics["accuracy"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a wafer defect classifier.")
    parser.add_argument("--data", default="data/sample_data.csv", help="Path to CSV data.")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for metrics, plots, and model artifacts.",
    )
    parser.add_argument("--test-size", default=0.25, type=float, help="Test split size.")
    parser.add_argument("--random-state", default=42, type=int, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_model(
        data_path=args.data,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("Training complete")
    for metric_name, value in summary.items():
        print(f"{metric_name}: {value:.3f}")


if __name__ == "__main__":
    main()
