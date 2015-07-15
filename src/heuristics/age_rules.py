# src/heuristics/age_rules.py
"""
age_rules.py

Age-based inclusion and exclusion rules for Trial Eligibility ML.

Purpose:
- Centralise age-related heuristic logic
- Make assumptions explicit and auditable
- Support later parameterisation via YAML (Phase B)

This module MUST:
- Use thresholds defined in config/thresholds.py
- Return boolean masks or flags only
- Contain no phenotype or modelling logic

This module MUST NOT:
- Derive age from dates
- Modify data in place
- Perform any I/O
"""

import pandas as pd

from config.thresholds import (
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

def apply_age_rules(df):
    """
    Apply age-based inclusion and exclusion rules.

    Parameters
    ----------
    df : pd.DataFrame
        Admission-level DataFrame containing age_at_admission.

    Returns
    -------
    pd.DataFrame
        DataFrame with age rule flags added:
        - age_in_range
        - age_excluded
    """

    required_columns = [
        "age_at_admission",
    ]

    require_columns(df, required_columns, context="Age rules input")
    require_numeric(df, "age_at_admission", context="Age rules input")

    df = df.copy()

    df["age_in_range"] = (
        (df["age_at_admission"] >= MIN_ELIGIBLE_AGE) &
        (df["age_at_admission"] <= MAX_ELIGIBLE_AGE)
    )

    df["age_excluded"] = df["age_at_admission"] > MAX_EXCLUSION_AGE

    return df


# ---------------------------------------------------------------------
# Convenience helpers (optional)
# ---------------------------------------------------------------------

def is_age_eligible(age):
    """
    Check whether a single age value passes inclusion rules.

    Parameters
    ----------
    age : float
        Age in years.

    Returns
    -------
    bool
        True if age is within eligible range and not excluded.
    """
    if age is None:
        return False

    if age > MAX_EXCLUSION_AGE:
        return False

    return MIN_ELIGIBLE_AGE <= age <= MAX_ELIGIBLE_AGE
