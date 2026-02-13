# Detailed Workflow

## Step 1: Data Import (`src/data/import_raw_data.py`)

- **Purpose**: Download raw data from AWS S3
- **Input**: None (downloads from S3)
- **Output**: 4 CSV files in `data/raw/`
  - `caracteristiques-2021.csv` (accident characteristics)
  - `lieux-2021.csv` (location data)
  - `usagers-2021.csv` (victim/user data)
  - `vehicules-2021.csv` (vehicle data)

## Step 2: Data Preprocessing (`src/data/make_dataset.py`)

- **Purpose**: Clean and merge raw data into a single dataset
- **Input**: 4 raw CSV files
- **Process**:
  - Merges all datasets on accident ID (`Num_Acc`)
  - Cleans data (handles missing values, converts types)
  - Creates aggregations (`nb_victim`, `nb_vehicules`)
  - Transforms target variable to binary classification
- **Output**: `data/preprocessed/interim_dataset.csv`

## Step 3: Feature Engineering (`src/features/build_features.py`)

- **Purpose**: Transform interim data into ML-ready features
- **Input**: `interim_dataset.csv`
- **Process**:
  - **Temporal features**: Creates datetime, extracts hour/month/day, cyclic encoding
  - **Age features**: Calculates victim age, creates age bins
  - **Categorical transformations**: Groups vehicle types, atmospheric conditions
  - **Interactions**: Creates feature interactions (e.g., `victims_per_vehicle`)
  - **Encoding**: Label encodes categorical features
- **Output**: 
  - `data/preprocessed/features.csv` (feature-engineered dataset)
  - `models/label_encoders.joblib` (saved encoders for inference)

## Step 4: Model Training (`src/models/train_multi_model.py`)

- **Purpose**: Train multiple models (XGBoost, Random Forest, Logistic Regression, LightGBM) with SMOTE for imbalanced data
- **Input**: `features.csv`
- **Process**:
  - Splits data into train/test sets (same split for all models for fair comparison)
  - Applies SMOTE (oversampling) to handle class imbalance
  - Trains multiple model types (with or without grid search)
  - Evaluates each model's performance
  - Generates model comparison report
- **Output**:
  - `models/{model_type}_model.joblib` (trained model pipelines, e.g., `xgboost_model.joblib`)
  - `models/{model_type}_model_metadata.joblib` (feature names, config per model)
  - `data/metrics/{model_type}_metrics.json` (evaluation metrics per model)
  - `data/metrics/model_comparison.csv` (comparison report ranking models by F1 score)

## Step 5: Prediction (`src/models/predict_model.py`)

- **Purpose**: Make predictions on new data
- **Input**: JSON file with input features
- **Process**:
  - Loads trained model and artifacts (encoders, metadata)
  - Preprocesses input using same pipeline as training (`src/features/preprocess.py`)
  - Aligns features with model expectations
  - Makes prediction
- **Output**: Prediction result (0 = Non-Priority, 1 = Priority)

## Complete Workflow Command

```bash
# Run entire pipeline in one command
make workflow-all

# Or run steps individually for more control
make run-import      # Step 1: Download raw data
make run-preprocess  # Step 2: Create interim dataset
make run-features    # Step 3: Build features
make run-train       # Step 4: Train model
make run-predict     # Step 5: Make predictions
```

## Reproducing the Workflow using DVC

Run `make dvc-repro` to reproduce the workflow using default configurations. This will:

1. Pull the latest version of raw data from the remote storage using `dvc pull`
2. Complete the workflow from Step 2 on

Note that DVC needs to be set up first using `make dvc-setup-remote`.

The default configurations are defined in [src/config/model_config.yaml](../src/config/model_config.yaml) and the prediction step is done on test features defined in [src/models/test_features.json](../src/models/test_features.json), which finally should output:

```
Prediction: 1
Interpretation: Priority
```

**Target (Post Phase 1)**: DVC Pipeline -> MLflow Tracking -> FastAPI Serving -> CI/CD
