# src/heuristics/stroke_rules.py
"""
stroke_rules.py

Stroke-signal inclusion rules for Trial Eligibility ML.

Purpose:
- Apply stroke-signal heuristic logic
- Read thresholds from study configuration where available
- Fall back to Phase A defaults when config is absent

This module MUST:
- Preserve Phase A behaviour by default
- Operate on admission-level phenotype features
- Return boolean flags only

This module MUST NOT:
- Derive phenotypes
- Perform any I/O
- Apply modelling logic
"""

import pandas as pd

from config.thresholds import (
    MIN_STROKE_CODE_COUNT,
    PREFER_PRIMARY_STROKE_DIAGNOSIS,
)
from utils.checks import (
    require_columns,
    require_numeric,
    require_boolean,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def apply_stroke_rules(df, study_config=None):
    """
    Apply stroke-signal inclusion rules.

    Parameters
    ----------
    df : pd.DataFrame
        Admission-level DataFrame containing stroke phenotype features.
    study_config : dict or None
        Optional study configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with stroke rule flags added:
        - stroke_signal_ok
        - stroke_primary_preferred
    """

    required_columns = [
        "stroke_code_count",
        "has_any_stroke_signal",
    ]

    require_columns(df, required_columns, context="Stroke rules input")
    require_numeric(df, "stroke_code_count", context="Stroke rules input")
    require_boolean(df, "has_any_stroke_signal", context="Stroke rules input")

    # Resolve thresholds (study config overrides defaults)
    if study_config and "stroke_signal" in study_config:
        stroke_cfg = study_config.get("stroke_signal", {})
        min_code_count = stroke_cfg.get(
            "min_code_count", MIN_STROKE_CODE_COUNT
        )
        require_any_signal = stroke_cfg.get(
            "require_any_signal", True
        )
        prefer_primary = stroke_cfg.get(
            "prefer_primary_dx", PREFER_PRIMARY_STROKE_DIAGNOSIS
        )
    else:
        min_code_count = MIN_STROKE_CODE_COUNT
        require_any_signal = True
        prefer_primary = PREFER_PRIMARY_STROKE_DIAGNOSIS

    df = df.copy()

    # Core stroke signal rule (permissive, recall-oriented)
    if require_any_signal:
        df["stroke_signal_ok"] = (
            df["has_any_stroke_signal"] &
            (df["stroke_code_count"] >= min_code_count)
        )
    else:
        df["stroke_signal_ok"] = (
            df["stroke_code_count"] >= min_code_count
        )

    # Primary diagnosis preference (informational only)
    if prefer_primary and "stroke_primary_dx_flag" in df.columns:
        df["stroke_primary_preferred"] = (
            df["stroke_primary_dx_flag"].astype(bool)
        )
    else:
        df["stroke_primary_preferred"] = False

    return df


# ---------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------

def is_stroke_signal_ok(
    stroke_code_count,
    has_any_stroke_signal,
    study_config=None,
):
    """
    Check whether stroke-signal criteria are satisfied for a single admission.

    Parameters
    ----------
    stroke_code_count : int
        Number of stroke-related diagnosis codes.
    has_any_stroke_signal : bool
        Whether any stroke-related signal exists.
    study_config : dict or None
        Optional study configuration.

    Returns
    -------
    bool
        True if stroke signal meets inclusion criteria.
    """

    if study_config and "stroke_signal" in study_config:
        stroke_cfg = study_config.get("stroke_signal", {})
        min_code_count = stroke_cfg.get(
            "min_code_count", MIN_STROKE_CODE_COUNT
        )
        require_any_signal = stroke_cfg.get(
            "require_any_signal", True
        )
    else:
        min_code_count = MIN_STROKE_CODE_COUNT
        require_any_signal = True

    if require_any_signal and not has_any_stroke_signal:
        return False

    return stroke_code_count >= min_code_count
