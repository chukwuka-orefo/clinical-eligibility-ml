# src/heuristics/age_rules.py
"""
age_rules.py

Age-based inclusion and exclusion rules for Trial Eligibility ML.

Purpose:
- Apply age-related heuristic logic
- Read thresholds from study configuration where available
- Fall back to Phase A defaults when config is absent

This module MUST:
- Use study config if provided
- Preserve Phase A behaviour by default
- Contain no phenotype or modelling logic

This module MUST NOT:
- Derive age from dates
- Modify data in place
- Perform any I/O
"""

import pandas as pd

from app.engine.config.thresholds import (
    MIN_ELIGIBLE_AGE,
    MAX_ELIGIBLE_AGE,
    MAX_EXCLUSION_AGE,
)
from utils.checks import (
    require_columns,
    require_numeric,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def apply_age_rules(df, study_config=None):
    """
    Apply age-based inclusion and exclusion rules.

    Parameters
    ----------
    df : pd.DataFrame
        Admission-level DataFrame containing age_at_admission.
    study_config : dict or None
        Optional study configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with age rule flags added:
        - age_in_range
        - age_excluded
    """

    require_columns(df, ["age_at_admission"], context="Age rules input")
    require_numeric(df, "age_at_admission", context="Age rules input")

    # Resolve thresholds (study config overrides defaults)
    if study_config and "age" in study_config:
        age_cfg = study_config.get("age", {})
        min_age = age_cfg.get("min", MIN_ELIGIBLE_AGE)
        max_age = age_cfg.get("max", MAX_ELIGIBLE_AGE)
        hard_exclude = age_cfg.get("hard_exclude", MAX_EXCLUSION_AGE)
    else:
        min_age = MIN_ELIGIBLE_AGE
        max_age = MAX_ELIGIBLE_AGE
        hard_exclude = MAX_EXCLUSION_AGE

    df = df.copy()

    df["age_in_range"] = (
        (df["age_at_admission"] >= min_age) &
        (df["age_at_admission"] <= max_age)
    )

    df["age_excluded"] = df["age_at_admission"] > hard_exclude

    return df


# ---------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------

def is_age_eligible(age, study_config=None):
    """
    Check whether a single age value passes inclusion rules.

    Parameters
    ----------
    age : float
        Age in years.
    study_config : dict or None
        Optional study configuration.

    Returns
    -------
    bool
        True if age is eligible.
    """

    if age is None:
        return False

    if study_config and "age" in study_config:
        age_cfg = study_config.get("age", {})
        min_age = age_cfg.get("min", MIN_ELIGIBLE_AGE)
        max_age = age_cfg.get("max", MAX_ELIGIBLE_AGE)
        hard_exclude = age_cfg.get("hard_exclude", MAX_EXCLUSION_AGE)
    else:
        min_age = MIN_ELIGIBLE_AGE
        max_age = MAX_ELIGIBLE_AGE
        hard_exclude = MAX_EXCLUSION_AGE

    if age > hard_exclude:
        return False

    return (age >= min_age) and (age <= max_age)
