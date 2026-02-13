# Pipeline Overview

The ML pipeline follows a simple 5-step workflow from raw data to predictions:

```
┌─────────────────┐
│  1. Data Import │  Downloads 4 CSV files from AWS S3
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Preprocessing│  Cleans & merges data → interim dataset
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Feature Eng. │  Creates ML-ready features (temporal, cyclic, interactions)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Training    │  Trains multiple models (XGBoost, RF, LR, LightGBM) with SMOTE
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Prediction  │  Makes predictions on new data
└─────────────────┘
```

## Key Files & Their Roles

| Step | File | What It Does | Output |
|------|------|--------------|--------|
| **1. Import** | `src/data/import_raw_data.py` | Downloads raw CSV files from S3 | `data/raw/*.csv` |
| **2. Preprocess** | `src/data/make_dataset.py` | Merges 4 datasets, cleans data, creates target variable | `data/preprocessed/interim_dataset.csv` |
| **3. Features** | `src/features/build_features.py` | Feature engineering: temporal, cyclic encoding, interactions | `data/preprocessed/features.csv` + `models/label_encoders.joblib` |
| **4. Train** | `src/models/train_multi_model.py` | Trains multiple models, compares performance, saves models + metadata | `models/{model_type}_model.joblib` + `data/metrics/model_comparison.csv` |
| **5. Predict** | `src/models/predict_model.py` | Loads model, preprocesses input, makes predictions | Prediction results |

## Supporting Utilities

- **`src/features/preprocess.py`**: Reusable preprocessing functions for inference (ensures training/inference consistency)

## Quick Command Reference

```bash
# Run complete pipeline
make workflow-all

# Or run steps individually
make run-import      # Step 1: Download data
make run-preprocess  # Step 2: Clean & merge
make run-features    # Step 3: Feature engineering
make run-train       # Step 4: Train model
make run-predict     # Step 5: Make predictions
```
