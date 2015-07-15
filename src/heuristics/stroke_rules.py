# src/heuristics/stroke_rules.py
"""
stroke_rules.py

Stroke-signal inclusion rules for Trial Eligibility ML.

Purpose:
- Centralise stroke-signal heuristic logic
- Make assumptions explicit and auditable
- Support later parameterisation via YAML (Phase B)

This module MUST:
- Use thresholds defined in config/thresholds.py
- Operate on admission-level phenotype features
- Return boolean flags only

This module MUST NOT:
- Derive phenotypes
- Modify thresholds dynamically
- Perform any I/O
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

def apply_stroke_rules(df):
    """
    Apply stroke-signal inclusion rules.

    Parameters
    ----------
    df : pd.DataFrame
        Admission-level DataFrame containing stroke phenotype features.

    Returns
    -------
    pd.DataFrame
        DataFrame with stroke rule flags added:
        - stroke_signal_ok
    """

    required_columns = [
        "stroke_code_count",
        "has_any_stroke_signal",
    ]

    require_columns(df, required_columns, context="Stroke rules input")
    require_numeric(df, "stroke_code_count", context="Stroke rules input")
    require_boolean(df, "has_any_stroke_signal", context="Stroke rules input")

    df = df.copy()

    # Basic stroke signal rule (permissive, recall-oriented)
    df["stroke_signal_ok"] = (
        df["has_any_stroke_signal"] &
        (df["stroke_code_count"] >= MIN_STROKE_CODE_COUNT)
    )

    # Optional preference for primary diagnosis (informational only)
    if PREFER_PRIMARY_STROKE_DIAGNOSIS and "stroke_primary_dx_flag" in df.columns:
        df["stroke_primary_preferred"] = df["stroke_primary_dx_flag"].astype(bool)
    else:
        df["stroke_primary_preferred"] = False

    return df


# ---------------------------------------------------------------------
# Convenience helpers (optional)
# ---------------------------------------------------------------------

def is_stroke_signal_ok(
    stroke_code_count,
    has_any_stroke_signal,
):
    """
    Check whether stroke-signal criteria are satisfied for a single admission.

    Parameters
    ----------
    stroke_code_count : int
        Number of stroke-related diagnosis codes.
    has_any_stroke_signal : bool
        Whether any stroke-related signal exists.

    Returns
    -------
    bool
        True if stroke signal meets inclusion criteria.
    """
    if not has_any_stroke_signal:
        return False

    return stroke_code_count >= MIN_STROKE_CODE_COUNT
