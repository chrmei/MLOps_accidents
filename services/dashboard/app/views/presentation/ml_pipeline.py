"""Data Science & ML Pipeline - Presentation Page (Christian - Detailed)."""

import streamlit as st


def render():
    """Render ML Pipeline page with detailed graphics and overviews."""
    st.title("Data Science & ML Pipeline")
    st.markdown("**Presenter: Christian**")
    st.markdown("---")

    # ML Pipeline Overview
    st.header("ML Pipeline - 5-Step Workflow")
    
    # Visual workflow diagram using columns
    col1, arrow1, col2, arrow2, col3, arrow3, col4, arrow4, col5 = st.columns([1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1])
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background-color: #2d2d3d; border-radius: 5px; border: 2px solid #e63946;">
            <h3>1️⃣</h3>
            <strong>Data Import</strong>
            <p style="font-size: 0.9em; margin-top: 10px;">Download from AWS S3<br/>(4 CSV files)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow1:
        st.markdown("<br><br><h2>→</h2>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background-color: #2d2d3d; border-radius: 5px; border: 2px solid #e63946;">
            <h3>2️⃣</h3>
            <strong>Preprocessing</strong>
            <p style="font-size: 0.9em; margin-top: 10px;">Clean, merge,<br/>create target</p>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow2:
        st.markdown("<br><br><h2>→</h2>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background-color: #2d2d3d; border-radius: 5px; border: 2px solid #e63946;">
            <h3>3️⃣</h3>
            <strong>Feature Engineering</strong>
            <p style="font-size: 0.9em; margin-top: 10px;">Temporal features,<br/>cyclic encoding, interactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow3:
        st.markdown("<br><br><h2>→</h2>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background-color: #2d2d3d; border-radius: 5px; border: 2px solid #e63946;">
            <h3>4️⃣</h3>
            <strong>Training</strong>
            <p style="font-size: 0.9em; margin-top: 10px;">Multi-model framework<br/>(XGBoost, RF, LR, LightGBM)<br/>with SMOTE</p>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow4:
        st.markdown("<br><br><h2>→</h2>", unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background-color: #2d2d3d; border-radius: 5px; border: 2px solid #e63946;">
            <h3>5️⃣</h3>
            <strong>Prediction</strong>
            <p style="font-size: 0.9em; margin-top: 10px;">Inference with<br/>preprocessing consistency</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Detailed Pipeline Steps
    with st.expander("📋 Detailed Pipeline Steps", expanded=True):
        st.subheader("Step 1: Data Import")
        st.code("""
# Download from AWS S3
- usagers.csv (victims/users)
- caracteristiques.csv (accident characteristics)
- lieux.csv (location data)
- vehicules.csv (vehicle information)
        """, language="python")
        
        st.subheader("Step 2: Preprocessing")
        st.code("""
# Clean and merge datasets
- Validate schemas
- Handle missing values
- Merge on accident ID
- Create target variable (severity)
        """, language="python")
        
        st.subheader("Step 3: Feature Engineering")
        st.code("""
# Temporal features
- Extract date components
- Cyclic encoding (sin/cos for time)
- Day of week, month, season

# Interactions
- Feature combinations
- Polynomial features
        """, language="python")
        
        st.subheader("Step 4: Training")
        st.code("""
# Multi-model framework
- XGBoost
- Random Forest
- Logistic Regression
- LightGBM

# Handle class imbalance
- SMOTE (Synthetic Minority Oversampling)
        """, language="python")
        
        st.subheader("Step 5: Prediction")
        st.code("""
# Inference pipeline
- Load model from registry
- Apply same preprocessing
- Predict severity
- Return probabilities
        """, language="python")

    st.markdown("---")

    # MLflow Integration
    st.header("MLflow Integration")
    
    # Model Registry Workflow Diagram
    st.subheader("Model Registry Workflow")
    
    registry_col1, arrow1, registry_col2, arrow2, registry_col3, arrow3, registry_col4 = st.columns([1, 0.1, 1, 0.1, 1, 0.1, 1])
    
    with registry_col1:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background-color: #1e1e2e; border-radius: 5px; border: 2px solid #e63946;">
            <h4>None</h4>
            <p>New Model<br/>Registered</p>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow1:
        st.markdown("<br><br><h2>↓</h2>", unsafe_allow_html=True)
    
    with registry_col2:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background-color: #1e1e2e; border-radius: 5px; border: 2px solid #ffa500;">
            <h4>Staging</h4>
            <p>Testing<br/>Validation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow2:
        st.markdown("<br><br><h2>↓</h2>", unsafe_allow_html=True)
    
    with registry_col3:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background-color: #1e1e2e; border-radius: 5px; border: 2px solid #00ff00;">
            <h4>Production</h4>
            <p>Live<br/>Deployment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow3:
        st.markdown("<br><br><h2>↓</h2>", unsafe_allow_html=True)
    
    with registry_col4:
        st.markdown("""
        <div style="text-align: center; padding: 15px; background-color: #1e1e2e; border-radius: 5px; border: 2px solid #808080;">
            <h4>Archived</h4>
            <p>Deprecated<br/>History</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("🔧 MLflow Features", expanded=True):
        st.markdown("""
        **Experiment Tracking:**
        - Metrics (accuracy, precision, recall, F1)
        - Parameters (hyperparameters, model config)
        - Artifacts (models, plots, reports)
        
        **Model Registry:**
        - Versioning: Each training run creates new version
        - Staging: None → Staging → Production → Archived
        - Remote tracking via DagsHub
        
        **Multi-Model Support:**
        - Separate registry entries per model type
        - Format: `Accident_Prediction_{ModelType}`
        - Examples: `Accident_Prediction_XGBoost`, `Accident_Prediction_RandomForest`
        """)

    st.markdown("---")

    # Model Management
    st.header("Model Management")
    
    mgmt_col1, mgmt_col2 = st.columns(2)
    
    with mgmt_col1:
        st.subheader("Config-Driven Training")
        st.code("""
# model_config.yaml
multi_model:
  enabled_models:
    - "xgboost"
    - "random_forest"
    - "logistic_regression"
    - "lightgbm"

mlflow:
  experiment_name: "accident_prediction"
  model_registry:
    registered_model_name: "Accident_Prediction"
        """, language="yaml")
    
    with mgmt_col2:
        st.subheader("Automatic Features")
        st.markdown("""
        ✅ **Model Comparison**
        - Automatic ranking by F1 score
        - Comparison report: `data/metrics/model_comparison.csv`
        
        ✅ **Best Model Selection**
        - Auto-select best performing model
        - Configurable via `--use-best-model` flag
        
        ✅ **Production Selection**
        - Load from Production stage
        - Fallback to local models
        """)

    st.markdown("---")

    # Data Versioning (Rafael's section - simple)
    st.header("Data Versioning")
    st.markdown(
        """
        - **DVC + DagsHub** for data pipeline reproducibility
        - **Pipeline stages**: preprocess → build_features → train_eval
        - Version control for datasets and artifacts
        """
    )
