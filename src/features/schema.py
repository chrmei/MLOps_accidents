# -*- coding: utf-8 -*-
"""
Canonical input feature schema definition.

This module defines the canonical input features that the frontend always sends
and that models expect as input before feature engineering. This schema NEVER changes,
ensuring backward compatibility and a stable API contract.

The frontend always collects these features, and the preprocessing pipeline
handles all transformations (grouping, encoding, interactions) automatically.
"""
from typing import Dict, List

# Canonical input features - this is the API contract that NEVER changes
# Frontend always sends these features, preprocessing handles transformations
CANONICAL_INPUT_FEATURES: List[str] = [
    # Victim/user features
    "place",
    "catu",
    "sexe",
    "secu1",
    "year_acc",  # Can be converted to 'an' during preprocessing
    "an_nais",
    # Vehicle features
    "catv",
    "obsm",
    "motor",
    # Location/road features
    "catr",
    "circ",
    "surf",
    "situ",
    "vma",
    # Temporal features
    "jour",
    "mois",
    "an",
    "hrmn",
    # Environmental features
    "lum",
    "dep",
    "com",
    "agg_",
    "int",
    "atm",
    "col",
    # Geographic coordinates
    "lat",
    "long",
    # Aggregated features (created during data preprocessing)
    "nb_victim",
    "nb_vehicules",
    # Pedestrian features (may be 0)
    "locp",
    "actp",
    "etatp",
    "obs",
    # Additional features
    "v1",
    "vosp",
    "prof",
    "plan",
    "larrout",
    "infra",
]

# Default values for optional features (used when features are missing)
CANONICAL_INPUT_DEFAULTS: Dict[str, any] = {
    "locp": 0,
    "actp": 0,
    "etatp": 0,
    "obs": 0,
    "v1": 0,
    "vosp": 0,
    "prof": 0,
    "plan": 0,
    "larrout": 0.0,
    "infra": 0,
}


def get_canonical_input_features() -> List[str]:
    """
    Get the list of canonical input features.
    
    Returns
    -------
    List[str]
        List of canonical input feature names
    """
    return CANONICAL_INPUT_FEATURES.copy()


def get_canonical_input_defaults() -> Dict[str, any]:
    """
    Get default values for optional canonical input features.
    
    Returns
    -------
    Dict[str, any]
        Dictionary mapping feature names to default values
    """
    return CANONICAL_INPUT_DEFAULTS.copy()


def validate_canonical_input(features: Dict[str, any]) -> Dict[str, any]:
    """
    Validate and normalize canonical input features.
    
    Ensures all required features are present and fills in defaults for optional features.
    
    Parameters
    ----------
    features : Dict[str, any]
        Input features dictionary
        
    Returns
    -------
    Dict[str, any]
        Validated and normalized features dictionary
    """
    validated = {}
    defaults = get_canonical_input_defaults()
    
    # Add all canonical features with defaults
    for feature in CANONICAL_INPUT_FEATURES:
        if feature in features:
            validated[feature] = features[feature]
        elif feature in defaults:
            validated[feature] = defaults[feature]
        else:
            # Required feature missing - will raise error during preprocessing
            validated[feature] = None
    
    return validated
