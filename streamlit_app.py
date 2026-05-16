"""Interactive dashboard for semiconductor wafer defect analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import TARGET_COLUMN, build_preprocessor, split_features_target


APP_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = APP_DIR / "data" / "sample_data.csv"


st.set_page_config(
    page_title="Semiconductor Defect Analysis",
    page_icon=None,
    layout="wide",
)


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH)


def load_uploaded_data(uploaded_file: Any | None) -> pd.DataFrame:
    if uploaded_file is None:
        return load_sample_data()
    return pd.read_csv(uploaded_file)


def train_dashboard_model(
    data: pd.DataFrame,
    test_size: float,
    n_estimators: int,
    random_state: int,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series, dict[str, float]]:
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
                    n_estimators=n_estimators,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    predictions = pd.Series(model.predict(x_test), index=x_test.index, name="prediction")
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
    }
    return model, x_test, y_test, predictions, metrics


def plot_class_distribution(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = data[TARGET_COLUMN].value_counts().index
    sns.countplot(data=data, x=TARGET_COLUMN, order=order, ax=ax, color="#2f80ed")
    ax.set_title("Defect Type Distribution")
    ax.set_xlabel("Defect type")
    ax.set_ylabel("Wafer count")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def plot_process_relationship(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        data=data,
        x="particle_count",
        y="surface_roughness_nm",
        hue=TARGET_COLUMN,
        s=80,
        ax=ax,
    )
    ax.set_title("Particle Count vs. Surface Roughness")
    ax.set_xlabel("Particle count")
    ax.set_ylabel("Surface roughness (nm)")
    ax.legend(title="Defect", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_test: pd.Series, predictions: pd.Series) -> plt.Figure:
    labels = sorted(set(y_test) | set(predictions))
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=labels,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def feature_importance_table(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{index}" for index in range(len(classifier.feature_importances_))]

    return (
        pd.DataFrame(
            {
                "feature": [name.split("__", 1)[-1] for name in feature_names],
                "importance": classifier.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def render_metric_cards(data: pd.DataFrame, metrics: dict[str, float]) -> None:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Wafers", f"{len(data):,}")
    metric_columns[1].metric("Defect Classes", data[TARGET_COLUMN].nunique())
    metric_columns[2].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    metric_columns[3].metric("Macro F1", f"{metrics['f1_macro']:.1%}")


def render_prediction_lab(model: Pipeline, data: pd.DataFrame) -> None:
    features, _ = split_features_target(data)

    wafer_labels = (
        data["wafer_id"].astype(str).tolist()
        if "wafer_id" in data.columns
        else [f"Row {index}" for index in data.index]
    )
    selected_label = st.selectbox("Wafer", wafer_labels)
    selected_position = wafer_labels.index(selected_label)
    selected_features = features.iloc[[selected_position]]

    prediction = model.predict(selected_features)[0]
    probabilities = None
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = pd.DataFrame(
            {
                "defect_type": model.classes_,
                "probability": model.predict_proba(selected_features)[0],
            }
        ).sort_values("probability", ascending=False)

    st.metric("Predicted Defect", str(prediction))
    if probabilities is not None:
        st.bar_chart(probabilities.set_index("defect_type"))

    with st.expander("Selected wafer features", expanded=False):
        st.dataframe(selected_features, width="stretch")


def main() -> None:
    st.title("Semiconductor Defect Analysis")
    st.caption("Process-quality dashboard for wafer defect classification.")

    with st.sidebar:
        st.header("Data and Model")
        uploaded_file = st.file_uploader("Upload wafer CSV", type=["csv"])
        test_size = st.slider("Test split", min_value=0.15, max_value=0.40, value=0.25, step=0.05)
        n_estimators = st.slider(
            "Random forest trees",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
        )
        random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

    data = load_uploaded_data(uploaded_file)
    if TARGET_COLUMN not in data.columns:
        st.error(f"CSV must include a '{TARGET_COLUMN}' column.")
        st.stop()

    try:
        model, x_test, y_test, predictions, metrics = train_dashboard_model(
            data=data,
            test_size=test_size,
            n_estimators=n_estimators,
            random_state=int(random_state),
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    render_metric_cards(data, metrics)

    overview_tab, model_tab, prediction_tab, data_tab = st.tabs(
        ["Overview", "Model Performance", "Prediction Lab", "Data"]
    )

    with overview_tab:
        left, right = st.columns(2)
        with left:
            st.pyplot(plot_class_distribution(data), width="stretch")
        with right:
            required_plot_columns = {"particle_count", "surface_roughness_nm", TARGET_COLUMN}
            if required_plot_columns.issubset(data.columns):
                st.pyplot(plot_process_relationship(data), width="stretch")
            else:
                st.info("Upload particle_count and surface_roughness_nm to show this chart.")

    with model_tab:
        left, right = st.columns([1, 1])
        with left:
            st.pyplot(plot_confusion_matrix(y_test, predictions), width="stretch")
        with right:
            st.subheader("Feature Importance")
            st.dataframe(feature_importance_table(model).head(12), width="stretch")

        st.subheader("Model Metrics")
        st.dataframe(
            pd.DataFrame([metrics]).rename(
                columns={
                    "accuracy": "Accuracy",
                    "precision_macro": "Precision (macro)",
                    "recall_macro": "Recall (macro)",
                    "f1_macro": "F1 (macro)",
                }
            ),
            width="stretch",
        )

    with prediction_tab:
        render_prediction_lab(model, data)

    with data_tab:
        st.subheader("Dataset Preview")
        st.dataframe(data, width="stretch")
        st.download_button(
            "Download current dataset",
            data.to_csv(index=False).encode("utf-8"),
            file_name="wafer_defect_data.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
