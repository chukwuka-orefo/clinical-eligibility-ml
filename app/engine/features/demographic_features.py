# src/features/demographic_features.py
"""
demographic_features.py

Build demographic features for Trial Eligibility ML.

Purpose:
- Provide a clean, explicit feature-layer interface for demographics
- Delegate to already-derived fields (no new logic)
- Keep feature construction consistent and auditable

This module MUST:
- Not derive age from raw timestamps
- Not introduce new clinical logic
- Not perform scaling or encoding

This module MUST NOT:
- Read raw data files
- Apply eligibility rules
- Perform model-specific transformations
"""

import pandas as pd

from utils.checks import (
    require_columns,
    require_non_null
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_demographic_features(df):
    """
    Build demographic features for modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Admission-level DataFrame containing demographic fields.

    Returns
    -------
    pd.DataFrame
        DataFrame with demographic feature columns only.
    """

    required_columns = [
        "hadm_id",
        "subject_id",
        "sex",
        "age_at_admission",
    ]

    require_columns(df, required_columns, context="Demographic feature input")

    # Age must already be derived upstream
    require_non_null(df, "age_at_admission", context="Demographic feature input")

    features = df[
        [
            "hadm_id",
            "subject_id",
            "sex",
            "age_at_admission",
        ]
    ].copy()

    # Normalise sex values (defensive, non-destructive)
    features["sex"] = (
        features["sex"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return features


# ---------------------------------------------------------------------
# Feature metadata (optional, for documentation / inspection)
# ---------------------------------------------------------------------

def get_demographic_feature_names():
    """
    Return the list of demographic feature column names.

    Returns
    -------
    List[str]
        Feature column names.
    """
    return [
        "age_at_admission",
        "sex",
    ]
