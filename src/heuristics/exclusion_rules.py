# src/heuristics/exclusion_rules.py
"""
exclusion_rules.py

Conservative exclusion rules for Trial Eligibility ML.

Purpose:
- Centralise exclusion logic that removes admissions from consideration
- Keep exclusions explicit, auditable, and conservative
- Support later parameterisation via YAML (Phase B)

This module MUST:
- Use thresholds defined in config/thresholds.py
- Operate at admission-level
- Return boolean exclusion flags only

This module MUST NOT:
- Apply inclusion logic
- Modify data in place
- Perform any I/O
"""

import pandas as pd

from config.thresholds import (
    MAX_EXCLUSION_AGE,
)
from utils.checks import (
    require_columns,
    require_numeric,
    require_boolean,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def apply_exclusion_rules(df):
    """
    Apply conservative exclusion rules.

    Parameters
    ----------
    df : pd.DataFrame
        Admission-level DataFrame containing required fields.

    Returns
    -------
    pd.DataFrame
        DataFrame with exclusion flags added:
        - excluded
        - exclusion_reason
    """

    required_columns = [
        "age_at_admission",
        "has_any_stroke_signal",
    ]

    require_columns(df, required_columns, context="Exclusion rules input")
    require_numeric(df, "age_at_admission", context="Exclusion rules input")
    require_boolean(df, "has_any_stroke_signal", context="Exclusion rules input")

    df = df.copy()

    # Initialise exclusion flags
    df["excluded"] = False
    df["exclusion_reason"] = ""

    # -----------------------------------------------------------------
    # Age-based hard exclusion
    # -----------------------------------------------------------------
    age_excluded = df["age_at_admission"] > MAX_EXCLUSION_AGE

    df.loc[age_excluded, "excluded"] = True
    df.loc[age_excluded, "exclusion_reason"] = "age_above_max_exclusion"

    # -----------------------------------------------------------------
    # No stroke signal exclusion (obvious non-candidates)
    # -----------------------------------------------------------------
    no_stroke_signal = ~df["has_any_stroke_signal"]

    df.loc[
        (~df["excluded"]) & no_stroke_signal,
        "excluded"
    ] = True

    df.loc[
        df["excluded"] & (df["exclusion_reason"] == "") & no_stroke_signal,
        "exclusion_reason"
    ] = "no_stroke_signal"

    return df


# ---------------------------------------------------------------------
# Convenience helpers (optional)
# ---------------------------------------------------------------------

def is_excluded(
    age_at_admission,
    has_any_stroke_signal,
):
    """
    Check whether an admission should be excluded based on core rules.

    Parameters
    ----------
    age_at_admission : float
        Age at admission.
    has_any_stroke_signal : bool
        Whether any stroke signal exists.

    Returns
    -------
    bool
        True if admission should be excluded.
    """
    if age_at_admission is None:
        return True

    if age_at_admission > MAX_EXCLUSION_AGE:
        return True

    if not has_any_stroke_signal:
        return True

    return False
