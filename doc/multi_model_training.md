# Multi-Model Training Framework

**The multi-model training framework is now the default training method.** The project includes a standardized framework for training and comparing multiple ML models. This framework enables easy experimentation with different algorithms (XGBoost, Random Forest, Logistic Regression, LightGBM) and automatic comparison of their performance.

## Features

- **Standardized Training**: All models use the same train/test split for fair comparison
- **MLflow Integration**: Models are automatically logged with type-specific tags for easy filtering
- **Model Registry**: Each model type gets its own registered model name (format: `Accident_Prediction_{ModelType}`)
- **Automatic Comparison**: Generates comparison reports ranking models by performance metrics
- **Extensible**: Easy to add new model types by creating a trainer class
- **Default Training**: Used by default in `make run-train` and DVC pipeline

## Quick Start

```bash
# Train all enabled models (default)
make run-train

# Train with grid search for hyperparameter tuning
make run-train-grid

# Train specific models only
python src/models/train_multi_model.py --models xgboost random_forest

# Train single model (legacy mode)
make run-train-single
```

## Output Files

After training, you'll find:
- **Models**: `models/{model_type}_model.joblib` (e.g., `models/xgboost_model.joblib`)
- **Metadata**: `models/{model_type}_model_metadata.joblib`
- **Metrics**: `data/metrics/{model_type}_metrics.json`
- **Comparison**: `data/metrics/model_comparison.csv` (ranks models by F1 score)

## Model Names in Registry

Models are registered with the format `Accident_Prediction_{ModelType}`:
- `Accident_Prediction_XGBoost`
- `Accident_Prediction_Random_Forest`
- `Accident_Prediction_Logistic_Regression`
- `Accident_Prediction_Lightgbm`

This naming convention groups all models under the project prefix for easy identification in MLflow.

For detailed documentation on how to add new models, see [src/models/README_MULTI_MODEL.md](../src/models/README_MULTI_MODEL.md).

## Quick Reference: Common Commands

```bash
# LIST MODELS
python scripts/manage_model_registry.py list-models
python scripts/manage_model_registry.py list-versions --model-name Accident_Prediction_XGBoost

# TRANSITION STAGES
python scripts/manage_model_registry.py transition --model-name Accident_Prediction_XGBoost --version 1 --stage Staging
python scripts/manage_model_registry.py transition --model-name Accident_Prediction_XGBoost --version 1 --stage Production

# PROMOTE LATEST VERSION
python scripts/manage_model_registry.py promote --model-name Accident_Prediction_XGBoost --stage Production

# ARCHIVE OLD MODEL
python scripts/manage_model_registry.py archive --model-name Accident_Prediction_XGBoost --version 1

# GET MODEL INFO
python scripts/manage_model_registry.py get-model --model-name Accident_Prediction_XGBoost --stage Production

# USE IN PREDICTIONS
python src/models/predict_model.py file.json --model-name Accident_Prediction_XGBoost --stage Production
python src/models/predict_model.py file.json --model-name Accident_Prediction_XGBoost --version 1
python src/models/predict_model.py file.json --model-path models/trained_model.joblib  # Local filesystem
```
