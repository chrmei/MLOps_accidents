# Automatic Feature Selection by Model Type and Version

The system implements **automatic feature selection** based on model metadata, ensuring the correct features are used without manual configuration. This enables:

- **Fixed Frontend API Contract**: Frontend always sends the same canonical input features (never changes)
- **Metadata-Driven Preprocessing**: Preprocessing pipeline automatically adapts based on model version
- **MLflow Signature Integration**: Models include MLflow signatures for automatic input validation
- **Backward Compatibility**: Old models work seamlessly via automatic metadata inference

## Canonical Input Schema

The frontend always sends the same **canonical input features** defined in `src/features/schema.py`. These features represent the raw input before any feature engineering:

- Victim/user features: `place`, `catu`, `sexe`, `secu1`, `year_acc`, `an_nais`
- Vehicle features: `catv`, `obsm`, `motor`
- Location/road features: `catr`, `circ`, `surf`, `situ`, `vma`
- Temporal features: `jour`, `mois`, `an`, `hrmn`
- Environmental features: `lum`, `dep`, `com`, `agg_`, `int`, `atm`, `col`
- Geographic coordinates: `lat`, `long`
- Aggregated features: `nb_victim`, `nb_vehicules`
- Additional features: `locp`, `actp`, `etatp`, `obs`, `v1`, `vosp`, `prof`, `plan`, `larrout`, `infra`

## Feature Engineering Configuration

Models store feature engineering configuration in metadata:

- **Feature Engineering Version**: e.g., `v2.0-grouped-features`
- **Grouped Features**: When enabled, categorical features are grouped to reduce cardinality
  - `place` -> `place_group`
  - `secu1` -> `secu_group`
  - `catv` -> `catv_group`
  - `motor` -> `motor_group`
  - `obsm` -> `obsm_group`
  - `obs` -> `obs_group`
- **Transformations**: Cyclic encoding, interactions, feature removal

## MLflow Signatures

Models are logged to MLflow with:

- **Input Signature**: Defines canonical input features (before preprocessing)
- **Output Signature**: Defines prediction output format
- **Input Example**: Example of canonical input format for documentation

This enables:
- Automatic input validation
- Self-documenting model interface
- MLflow serving integration
- Better error messages

## API Endpoints

The predict service exposes model information endpoints:

- **`GET /api/v1/predict/model-info`**: Get detailed model information including feature engineering config
- **`GET /api/v1/predict/input-features`**: Get canonical input features expected by the model

Example response from `/api/v1/predict/model-info`:

```json
{
  "model_type": "XGBoost",
  "input_features": ["place", "secu1", "catv", "..."],
  "feature_engineering_config": {
    "feature_engineering_version": "v2.0-grouped-features",
    "uses_grouped_features": true,
    "grouped_feature_mappings": {
      "place": "place_group",
      "secu1": "secu_group"
    },
    "apply_cyclic_encoding": true,
    "apply_interactions": true
  },
  "mlflow_signature_available": true
}
```

## Workflow

**Data Scientist (Training)**:
1. Train model with feature engineering
2. System automatically captures canonical input features and feature engineering config
3. Model logged to MLflow with signature and metadata
4. No frontend/API changes needed

**MLOps Engineer (Serving)**:
1. Load model from MLflow registry
2. System automatically reads signature and metadata
3. Preprocessing pipeline adapts based on metadata
4. Input validation via MLflow signature

**Backward Compatibility**:
- Old models without metadata: Config inferred from feature names
- Old models without signatures: Fallback to metadata or canonical schema
- Graceful degradation ensures all models work
