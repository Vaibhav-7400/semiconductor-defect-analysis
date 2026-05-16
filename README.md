# Semiconductor Defect Analysis

Interactive machine learning project for detecting and classifying semiconductor wafer defects from process and inspection data.

## Project Summary

Semiconductor manufacturing depends on early defect detection to protect yield, reduce scrap, and identify process drift before it reaches production scale. This project builds a supervised machine learning workflow that classifies wafer defect patterns such as scratches, particle defects, edge-ring issues, contamination, and normal wafers.

The repository includes a training pipeline, synthetic sample wafer data, evaluation outputs, and a Streamlit dashboard for exploring the dataset, training a baseline Random Forest model, reviewing performance, and testing single-wafer predictions.

## Live Demo

[Open the deployed Streamlit dashboard](https://semiconductor-defect-analysis-m9ojmtnsjbyietv9sae4z8.streamlit.app/)

## Key Features

- Baseline Random Forest classifier for multi-class defect prediction
- Data preprocessing for numeric and categorical wafer features
- Model evaluation with accuracy, precision, recall, macro F1, and confusion matrix
- Feature importance analysis to highlight likely process drivers
- Streamlit dashboard for upload, exploration, training, and prediction
- Starter EDA notebook for exploratory analysis

## Tech Stack

- Python 3.12.4 for local development
- pandas and NumPy for data preparation
- scikit-learn Random Forest classifier for modeling
- matplotlib and seaborn for visualization
- Streamlit for the dashboard
- joblib for model persistence

## Project Structure

```text
semiconductor-defect-analysis/
|-- data/
|   `-- sample_data.csv
|-- notebooks/
|   `-- exploratory_analysis.ipynb
|-- results/
|   `-- .gitkeep
|-- src/
|   |-- __init__.py
|   |-- data_preprocessing.py
|   |-- evaluation.py
|   `-- model_training.py
|-- streamlit_app.py
|-- requirements.txt
`-- README.md
Quick Start
Create and activate a virtual environment:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Train the baseline model:

python -m src.model_training --data data/sample_data.csv --output-dir results
Run the dashboard:

streamlit run streamlit_app.py
If streamlit is not available as a direct command, use:

python -m streamlit run streamlit_app.py
Dashboard Workflow
Load the included synthetic wafer dataset or upload your own CSV.
Review class balance and process-feature relationships.
Train a Random Forest model from the sidebar controls.
Inspect accuracy, macro F1, confusion matrix, and feature importance.
Select an individual wafer in the Prediction Lab to view its predicted defect class.
Dataset
The included dataset is synthetic and intended for project development. Replace data/sample_data.csv with real process, metrology, or inspection data before using this workflow for manufacturing decisions.

Expected target column:

defect_type
Example feature columns:

temperature_c
pressure_torr
etch_rate_nm_min
deposition_thickness_nm
surface_roughness_nm
particle_count
line_width_variation_nm
voltage_v
current_a
lot_id
Results
The training command writes generated artifacts to results/:

defect_classifier.joblib
metrics.json
confusion_matrix.png
Generated model outputs are ignored by Git so the repository stays lightweight.

Future Enhancements
Add real wafer inspection or process-control data
Add cross-validation and hyperparameter tuning
Track model experiments and dataset versions
Add wafer-map image classification with CNN models
Add dashboard screenshots and deployment notes
Author
Vaibhav Krishna Naik

License
MIT License
