# Semiconductor Defect Analysis

Machine learning starter project for detecting and classifying semiconductor wafer defects using Python and scikit-learn.

## Project Overview

This project trains a supervised model on wafer process and inspection features to classify defect patterns such as scratches, particle defects, edge-ring issues, contamination, and normal wafers. It is designed as a clean baseline that can be extended with real manufacturing data, richer feature engineering, and image-based analysis.

## Technologies

- Python 3.10+
- pandas and NumPy for data preparation
- scikit-learn for modeling
- matplotlib and seaborn for evaluation visuals
- joblib for saving trained models

## Project Structure

```text
semiconductor-defect-analysis/
├── data/
│   └── sample_data.csv
├── notebooks/
│   └── exploratory_analysis.ipynb
├── results/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   └── model_training.py
├── requirements.txt
└── README.md
```

## Quick Start

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train the baseline model with the included sample data:

```powershell
python -m src.model_training --data data/sample_data.csv --output-dir results
```

After training, the `results/` folder will contain:

- `defect_classifier.joblib`
- `metrics.json`
- `confusion_matrix.png`

## Dataset

The sample dataset is synthetic and meant for development only. Replace `data/sample_data.csv` with real process, metrology, or inspection data before drawing manufacturing conclusions.

Expected target column:

- `defect_type`

Example feature columns:

- `temperature_c`
- `pressure_torr`
- `etch_rate_nm_min`
- `deposition_thickness_nm`
- `surface_roughness_nm`
- `particle_count`
- `line_width_variation_nm`
- `voltage_v`
- `current_a`
- `lot_id`

## Next Enhancements

- Add real wafer inspection data
- Expand feature engineering for process drift and tool-level signals
- Add cross-validation and hyperparameter tuning
- Add CNN-based image classification for wafer maps or microscope images
- Build a dashboard for model results and manufacturing quality trends

## Author

Vaibhav Krishna Naik

## License

MIT License
