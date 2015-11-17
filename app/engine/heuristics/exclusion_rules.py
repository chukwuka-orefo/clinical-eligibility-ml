# src/heuristics/exclusion_rules.py
"""
exclusion_rules.py

Conservative exclusion rules for Trial Eligibility ML.

Purpose:
- Centralise exclusion logic
- Read exclusion toggles from study configuration where available
- Fall back to Phase A defaults when config is absent

This module MUST:
- Preserve Phase A behaviour by default
- Operate at admission-level
- Return boolean exclusion flags only

This module MUST NOT:
- Apply inclusion logic
- Perform any I/O
- Modify global state
"""

import pandas as pd

from app.engine.config.thresholds import (
    MAX_EXCLUSION_AGE,
)
from app.engine.utils.checks import (
    require_columns,
    require_numeric,
    require_boolean,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def apply_exclusion_rules(df, study_config=None):
    """
    Apply conservative exclusion rules.

    Parameters
    ----------
    df : pd.DataFrame
        Admission-level DataFrame containing required fields.
    study_config : dict or None
        Optional study configuration.

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

    # Resolve exclusion toggles (study config overrides defaults)
    if study_config and "exclusions" in study_config:
        excl_cfg = study_config.get("exclusions", {})
        exclude_without_stroke = excl_cfg.get(
            "exclude_without_stroke_signal", True
        )
        exclude_if_age_above = excl_cfg.get(
            "exclude_if_age_above_hard_limit", True
        )
        hard_exclude_age = study_config.get("age", {}).get(
            "hard_exclude", MAX_EXCLUSION_AGE
        )
    else:
        exclude_without_stroke = True
        exclude_if_age_above = True
        hard_exclude_age = MAX_EXCLUSION_AGE

    df = df.copy()

    # Initialise exclusion flags
    df["excluded"] = False
    df["exclusion_reason"] = ""

    # Age-based hard exclusion
    if exclude_if_age_above:
        age_excluded = df["age_at_admission"] > hard_exclude_age

        df.loc[age_excluded, "excluded"] = True
        df.loc[age_excluded, "exclusion_reason"] = "age_above_hard_limit"

    # No stroke signal exclusion
    if exclude_without_stroke:
        no_stroke_signal = ~df["has_any_stroke_signal"]

        df.loc[
            (~df["excluded"]) & no_stroke_signal,
            "excluded"
        ] = True

        df.loc[
            (df["excluded"]) &
            (df["exclusion_reason"] == "") &
            no_stroke_signal,
            "exclusion_reason"
        ] = "no_stroke_signal"

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

    if study_config and "exclusions" in study_config:
        excl_cfg = study_config.get("exclusions", {})
        exclude_without_stroke = excl_cfg.get(
            "exclude_without_stroke_signal", True
        )
        exclude_if_age_above = excl_cfg.get(
            "exclude_if_age_above_hard_limit", True
        )
        hard_exclude_age = study_config.get("age", {}).get(
            "hard_exclude", MAX_EXCLUSION_AGE
        )
    else:
        exclude_without_stroke = True
        exclude_if_age_above = True
        hard_exclude_age = MAX_EXCLUSION_AGE

    if exclude_if_age_above and age_at_admission > hard_exclude_age:
        return True

    if exclude_without_stroke and not has_any_stroke_signal:
        return True

    return False
