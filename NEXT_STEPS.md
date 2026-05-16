# Next Steps

## Goal

Make the semiconductor defect analysis project more realistic by evaluating the model on a public wafer or semiconductor manufacturing dataset.

## Dataset Options

- WM-811K wafer map dataset
  - Best-known public wafer-map defect dataset
  - Good for semiconductor-specific pattern classification
  - Likely requires an image/array-based pipeline rather than the current tabular CSV-only pipeline

- MixedWM38 / mixed-type wafer defect datasets
  - Useful for single and mixed wafer defect pattern classification
  - Strong candidate if the project should show more advanced defect pattern handling

- SECOM semiconductor manufacturing dataset
  - Better match for tabular process/manufacturing features
  - Usually framed as binary good/defective classification rather than multi-class defect type classification

## Recommended Direction

Keep the current Random Forest model as the baseline tabular ML workflow. Add a second, more realistic dataset path:

```text
real wafer dataset -> preprocessing -> model training -> evaluation -> dashboard comparison
```

If using WM-811K or MixedWM38, add wafer-map loading and either:

- feature extraction plus classical ML, or
- a CNN-based image/array classifier

If using SECOM, adapt the current tabular pipeline for binary classification and compare it against the synthetic demo data.

## Tomorrow's First Task

Pick one public dataset, download it, and place it under:

```text
data/raw/
```

Then update the project with:

- dataset loading script
- real-data preprocessing
- model training comparison
- dashboard section for real dataset results
- README notes explaining that synthetic data is only the demo baseline
