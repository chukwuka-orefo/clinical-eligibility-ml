# src/features/phenotype_features.py
"""
phenotype_features.py

Build phenotype-based features for Trial Eligibility ML.

Purpose:
- Provide a clean feature-layer interface for clinical phenotypes
- Delegate to already-derived phenotype tables
- Make explicit which phenotype signals are model inputs

This module MUST:
- Not derive phenotypes itself
- Not apply eligibility rules
- Not perform feature scaling or encoding

This module MUST NOT:
- Read raw diagnosis data
- Recompute codelist logic
- Introduce new clinical assumptions
"""

import pandas as pd

from utils.checks import (
    require_columns,
    require_boolean,
    require_numeric,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_phenotype_features(
    df,
):
    """
    Build phenotype-based features for modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Admission-level DataFrame containing phenotype fields.

    Returns
    -------
    pd.DataFrame
        DataFrame with phenotype feature columns only.
    """

    required_columns = [
        "hadm_id",
        "stroke_code_count",
        "stroke_code_density",
        "has_any_stroke_signal",
        "stroke_primary_dx_flag",
        "cardiovascular_code_count",
        "cardiovascular_code_density",
        "has_any_cvd_signal",
    ]

    require_columns(df, required_columns, context="Phenotype feature input")

    # Type safety (defensive, non-transformative)
    require_numeric(df, "stroke_code_count", context="Phenotype feature input")
    require_numeric(df, "stroke_code_density", context="Phenotype feature input")
    require_boolean(df, "has_any_stroke_signal", context="Phenotype feature input")

    require_numeric(df, "cardiovascular_code_count", context="Phenotype feature input")
    require_numeric(df, "cardiovascular_code_density", context="Phenotype feature input")
    require_boolean(df, "has_any_cvd_signal", context="Phenotype feature input")

    features = df[
        [
            "hadm_id",
            "stroke_code_count",
            "stroke_code_density",
            "has_any_stroke_signal",
            "stroke_primary_dx_flag",
            "cardiovascular_code_count",
            "cardiovascular_code_density",
            "has_any_cvd_signal",
        ]
    ].copy()

    return features


# ---------------------------------------------------------------------
# Feature metadata (optional)
# ---------------------------------------------------------------------

def get_phenotype_feature_names():
    """
    Return the list of phenotype feature column names.

    Returns
    -------
    List[str]
        Feature column names.
    """
    return [
        "stroke_code_count",
        "stroke_code_density",
        "has_any_stroke_signal",
        "stroke_primary_dx_flag",
        "cardiovascular_code_count",
        "cardiovascular_code_density",
        "has_any_cvd_signal",
    ]
