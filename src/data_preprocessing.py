"""Data loading and preprocessing helpers for wafer defect classification."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "defect_type"
IGNORED_COLUMNS = {"wafer_id"}


def load_wafer_data(path: str | Path) -> pd.DataFrame:
    """Load wafer data from a CSV file and validate the target column."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    data = pd.read_csv(csv_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must include a '{TARGET_COLUMN}' column.")

    return data


def split_features_target(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    ignored_columns: Iterable[str] = IGNORED_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into model features and the target label."""
    ignored = set(ignored_columns)
    feature_columns = [
        column
        for column in data.columns
        if column != target_column and column not in ignored
    ]

    if not feature_columns:
        raise ValueError("No feature columns are available for training.")

    return data[feature_columns], data[target_column]


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing steps for numeric and categorical columns."""
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable numeric or categorical columns were found.")

    return ColumnTransformer(transformers=transformers)


def _one_hot_encoder() -> OneHotEncoder:
    """Create an encoder that works across recent scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
