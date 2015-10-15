# src/features/admission_features.py
"""
admission_features.py

Build admission-context features for Trial Eligibility ML.

Purpose:
- Provide a clean, explicit feature-layer interface for admission context
- Delegate to fields already derived during ingestion
- Formalise which admission fields are considered model inputs

This module MUST:
- Not derive new admission logic
- Not introduce temporal leakage
- Not perform feature scaling or encoding

This module MUST NOT:
- Read raw data files
- Apply clinical meaning to diagnoses
- Apply eligibility or modelling logic
"""

import pandas as pd

from utils.checks import (
    require_columns,
    require_non_null,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_admission_features(df):
    """
    Build admission-context features for modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Admission-level DataFrame containing admission fields.

    Returns
    -------
    pd.DataFrame
        DataFrame with admission-context feature columns only.
    """

    required_columns = [
        "hadm_id",
        "subject_id",
        "admission_type",
        "length_of_stay_days",
    ]

    require_columns(df, required_columns, context="Admission feature input")

    # length_of_stay_days may be missing for malformed records
    # Do not fail hard; allow NaN and handle downstream
    features = df[
        [
            "hadm_id",
            "subject_id",
            "admission_type",
            "length_of_stay_days",
        ]
    ].copy()

    # Normalise admission type defensively
    features["admission_type"] = (
        features["admission_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return features


# ---------------------------------------------------------------------
# Feature metadata (optional)
# ---------------------------------------------------------------------

def get_admission_feature_names():
    """
    Return the list of admission-context feature column names.

    Returns
    -------
    List[str]
        Feature column names.
    """
    return [
        "admission_type",
        "length_of_stay_days",
    ]
