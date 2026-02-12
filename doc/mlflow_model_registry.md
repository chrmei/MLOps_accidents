# MLflow Model Registry

The project uses MLflow Model Registry for model versioning and staging. Models are automatically registered during training, and you can manage their lifecycle through staging transitions.

## Configuration

Model registry settings are configured in `src/config/model_config.yaml`:

```yaml
mlflow:
  enabled: true
  tracking_uri: ""  # Set via MLFLOW_TRACKING_URI or DAGSHUB_REPO env vars
  experiment_name: "accident_prediction"
  log_model: true
  log_artifacts: true
  model_registry:
    registered_model_name: "Accident_Prediction"  # Base name - model type will be appended automatically
    default_stage: "None"  # Options: None, Staging, Production, Archived
    auto_transition_to_staging: false  # Auto-promote to Staging after registration
    production_stage: "Production"
```

## Quick Start: Model Staging Workflow

Follow these steps to train, stage, and deploy models using the MLflow Model Registry:

### Step 1: Set Up Environment

Configure your MLflow tracking URI (choose one method):

```bash
# Option 1: Set environment variable directly
export MLFLOW_TRACKING_URI="https://dagshub.com/yourusername/yourrepo.mlflow"

# Option 2: Use DAGSHUB_REPO (auto-constructs URI)
export DAGSHUB_REPO="yourusername/yourrepo"
```

Or add to your `.env` file for persistence.

### Step 2: Train and Register Models

Train models - they will be automatically registered:

```bash
# Train multiple models (default - creates versions for each model type)
make run-train

# Or with grid search for hyperparameter tuning
make run-train-grid

# Or train specific models only
python src/models/train_multi_model.py --models xgboost random_forest
```

**What happens:**
- Multiple models are trained (XGBoost, Random Forest, Logistic Regression, LightGBM by default)
- Each model is saved locally to `models/{model_type}_model.joblib`
- Each model is automatically registered to MLflow Model Registry with name `Accident_Prediction_{ModelType}`
- New versions are created for each model (starts in "None" stage)
- Metrics, parameters, and artifacts are logged to MLflow for each model
- Model comparison report is generated at `data/metrics/model_comparison.csv`

### Step 3: Check Registered Models

View what's in your registry:

```bash
# See all registered models
python scripts/manage_model_registry.py list-models

# See all versions of your specific model
python scripts/manage_model_registry.py list-versions \
  --model-name Accident_Prediction_XGBoost
```

### Step 4: Move Model to Staging

After validating the model locally, promote it to Staging for testing:

```bash
# Move version 1 to Staging
python scripts/manage_model_registry.py transition \
  --model-name Accident_Prediction_XGBoost \
  --version 1 \
  --stage Staging
```

### Step 5: Test Model from Staging

Test the Staging model to ensure it works correctly:

```bash
# Make predictions using Staging model
python src/models/predict_model.py src/models/test_features.json \
  --model-name Accident_Prediction_XGBoost \
  --stage Staging
```

### Step 6: Promote to Production

Once testing passes, promote to Production:

```bash
# Option A: Promote latest version (recommended)
python scripts/manage_model_registry.py promote \
  --model-name Accident_Prediction_XGBoost \
  --stage Production

# Option B: Promote specific version
python scripts/manage_model_registry.py transition \
  --model-name Accident_Prediction_XGBoost \
  --version 1 \
  --stage Production
```

### Step 7: Use Production Model

In production environments, always load from the Production stage:

```bash
# Load from Production stage (recommended)
python src/models/predict_model.py src/models/test_features.json \
  --model-name Accident_Prediction_XGBoost \
  --stage Production
```

### Step 8: Archive Old Models

When a model is deprecated, archive it (don't delete - maintains history):

```bash
# Archive old version
python scripts/manage_model_registry.py archive \
  --model-name Accident_Prediction_XGBoost \
  --version 1
```

## Complete Example Workflow

Here's a complete example from training to production:

```bash
# 1. Set up environment
export DAGSHUB_REPO="yourusername/yourrepo"

# 2. Train a new model
make run-train
# Output: Model registered as 'Accident_Prediction_XGBoost' version 1

# 3. Check what was registered
python scripts/manage_model_registry.py list-versions \
  --model-name Accident_Prediction_XGBoost
# You'll see version 1 in "None" stage

# 4. Move to Staging for testing
python scripts/manage_model_registry.py transition \
  --model-name Accident_Prediction_XGBoost \
  --version 1 \
  --stage Staging

# 5. Test the Staging model
python src/models/predict_model.py src/models/test_features.json \
  --model-name Accident_Prediction_XGBoost \
  --stage Staging

# 6. If tests pass, promote to Production
python scripts/manage_model_registry.py transition \
  --model-name Accident_Prediction_XGBoost \
  --version 1 \
  --stage Production

# 7. Use Production model
python src/models/predict_model.py src/models/test_features.json \
  --model-name Accident_Prediction_XGBoost \
  --stage Production

# 8. Later, train a better model (creates version 2)
make run-train

# 9. After validating version 2, promote it
python scripts/manage_model_registry.py promote \
  --model-name Accident_Prediction_XGBoost \
  --stage Production
# This moves version 2 to Production

# 10. Archive old version 1
python scripts/manage_model_registry.py archive \
  --model-name Accident_Prediction_XGBoost \
  --version 1
```

## Model Staging Lifecycle

Models progress through the following stages:

1. **None** (default) - Newly registered models start here
2. **Staging** - Models under evaluation/testing
3. **Production** - Models deployed and serving predictions
4. **Archived** - Deprecated models

## Managing Models

Use the `scripts/manage_model_registry.py` script to manage models:

### List Registered Models

```bash
# List all registered models
python scripts/manage_model_registry.py list-models

# List all versions of a specific model
python scripts/manage_model_registry.py list-versions --model-name Accident_Prediction_XGBoost
```

### Transition Models Between Stages

```bash
# Transition a specific version to Production
python scripts/manage_model_registry.py transition \
  --model-name Accident_Prediction_XGBoost \
  --version 1 \
  --stage Production

# Promote latest version to Production
python scripts/manage_model_registry.py promote \
  --model-name Accident_Prediction_XGBoost \
  --stage Production

# Archive an old version
python scripts/manage_model_registry.py archive \
  --model-name Accident_Prediction_XGBoost \
  --version 1
```

### Get Model Information

```bash
# Get model info by stage
python scripts/manage_model_registry.py get-model \
  --model-name Accident_Prediction_XGBoost \
  --stage Production
```

## Loading Models from Registry

The prediction script (`src/models/predict_model.py`) supports loading models from the MLflow registry:

**Best Practice: Use MLflow Model Registry for Production Inference**

```bash
# Automatically use best Production model across all model types (recommended)
python src/models/predict_model.py src/models/test_features.json \
  --use-best-model

# Or load from Production stage for specific model type
python src/models/predict_model.py src/models/test_features.json \
  --use-mlflow-production

# Or explicitly specify model name and stage
python src/models/predict_model.py src/models/test_features.json \
  --model-name Accident_Prediction_XGBoost \
  --stage Production

# Load specific version
python src/models/predict_model.py src/models/test_features.json \
  --model-name Accident_Prediction_XGBoost \
  --version 6

# Load from local filesystem (for development/testing only)
python src/models/predict_model.py src/models/test_features.json \
  --model-path models/xgboost_model.joblib

# Use environment variables
export USE_BEST_MODEL=true  # Auto-select best model
# or
export USE_MLFLOW_PRODUCTION=true  # Use default model type (XGBoost)
python src/models/predict_model.py src/models/test_features.json
```

**Architecture:**
- **MLflow Model Registry**: Used for production model serving (default with `--use-mlflow-production`)
- **Local filesystem (DVC)**: Used for development/testing and pipeline reproducibility
- Models are tracked in DVC for reproducible training pipelines, but production inference loads from MLflow

## Automatic Staging Transitions

You can enable automatic transition to Staging after model registration by setting `auto_transition_to_staging: true` in the config. This is useful for automated workflows where new models should be immediately available for testing.

## Model Storage Architecture (Best Practices)

**MLflow Model Registry** (for models):
- Production model serving and deployment
- Model versioning and lifecycle management (Staging -> Production -> Archived)
- Model metadata, metrics, and parameters tracking
- Experiment tracking and model comparison

**DVC** (for data):
- Data pipeline reproducibility (raw -> preprocessed -> features)
- Data versioning and tracking
- Label encoders and preprocessing artifacts
- Metrics files for pipeline tracking
- Model files tracked only for pipeline reproducibility (not for production use)

**Key Points:**
- Production inference should load from MLflow Model Registry (Production stage)
- **Multi-model setup**: Use `--use-best-model` to automatically select the best performing Production model across all model types (XGBoost, RandomForest, etc.)
- Each model type is registered separately: `Accident_Prediction_XGBoost`, `Accident_Prediction_Random_Forest`, etc.
- Local model files in DVC are for development/testing and pipeline reproducibility only
- Use `--use-mlflow-production` flag or `USE_MLFLOW_PRODUCTION=true` for production inference (defaults to XGBoost)
- Use `--use-best-model` flag or `USE_BEST_MODEL=true` to auto-select best model
- `make dvc-pull` pulls data and training artifacts for local development, not production models

## Best Practices

1. **Version Control**: Each training run creates a new model version automatically
2. **Staging Workflow**: 
   - New models -> None stage
   - After validation -> Staging stage
   - After approval -> Production stage
3. **Production Models**: Always load from Production stage in production environments
4. **Archiving**: Archive old models instead of deleting them to maintain history

## Environment Setup

Set up MLflow tracking URI via environment variables:

```bash
# Option 1: Direct MLflow URI
export MLFLOW_TRACKING_URI="https://dagshub.com/username/repo.mlflow"

# Option 2: Use DAGSHUB_REPO (auto-constructs URI)
export DAGSHUB_REPO="username/repo"
```

The tracking URI can also be set in `model_config.yaml`, but environment variables take precedence.
