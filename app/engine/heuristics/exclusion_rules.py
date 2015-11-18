# src/heuristics/exclusion_rules.py
"""
exclusion_rules.py

Apply exclusion heuristics for Trial Eligibility ML.

Purpose:
- Apply conservative exclusion logic
- Respect study configuration where available
- Preserve Phase A behaviour when data is present

This module MUST:
- Operate at admission (hadm_id) level
- Be transparent and auditable
- Avoid over-exclusion

This module MUST NOT:
- Train models
- Perform feature scaling
- Derive new features
"""

import pandas as pd

from app.engine.config.thresholds import (
    MAX_EXCLUSION_AGE,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def apply_exclusion_rules(df, study_config=None):
    """
    Apply exclusion heuristics.

    Parameters
    ----------
    df : pd.DataFrame
        Admission-level DataFrame.
    study_config : dict or None
        Optional study configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with exclusion flags applied.
    """

    df = df.copy()

    # -------------------------------------------------------------
    # Resolve exclusion threshold
    # -------------------------------------------------------------
    if study_config and "age" in study_config:
        hard_exclude = study_config.get("age", {}).get(
            "hard_exclude",
            MAX_EXCLUSION_AGE,
        )
    else:
        hard_exclude = MAX_EXCLUSION_AGE

    # -------------------------------------------------------------
    # Initialise exclusion flag if not present
    # -------------------------------------------------------------
    if "excluded" not in df.columns:
        df["excluded"] = False

    # -------------------------------------------------------------
    # Age-based exclusion (only if age is available)
    # -------------------------------------------------------------
    if "age_at_admission" in df.columns:
        df["excluded"] = df["excluded"] | (
            df["age_at_admission"] > hard_exclude
        )

    return df


# ---------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------

def is_excluded(
    age_at_admission,
    has_any_stroke_signal,
    study_config=None,
):
    """
    Check whether an admission should be excluded.

    Parameters
    ----------
    age_at_admission : float
        Age at admission.
    has_any_stroke_signal : bool
        Whether any stroke signal exists.
    study_config : dict or None
        Optional study configuration.

    Returns
    -------
    bool
        True if admission should be excluded.
    """

    if age_at_admission is None:
        return False

    if study_config and "age" in study_config:
        hard_exclude = study_config.get("age", {}).get(
            "hard_exclude",
            MAX_EXCLUSION_AGE,
        )
    else:
        hard_exclude = MAX_EXCLUSION_AGE

    if age_at_admission > hard_exclude:
        return True

    return False
