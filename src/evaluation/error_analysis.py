# src/evaluation/error_analysis.py
"""
error_analysis.py

Error analysis utilities for Trial Eligibility ML.

Purpose:
- Identify and inspect false negatives and false positives
- Support clinical and analytical review of screening behaviour
- Produce tabular outputs suitable for CSV export and discussion

This module MUST:
- Operate on scored admission-level data
- Treat heuristic labels as noisy reference signals
- Produce human-readable outputs

This module MUST NOT:
- Train models
- Modify scores or labels
- Perform visualisation
"""

import pandas as pd

from utils.checks import (
    require_columns,
    require_boolean,
    require_probability,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def analyse_errors(
    df,
    score_threshold,
):
    """
    Analyse false positives and false negatives at a given score threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Scored admissions table containing:
        - eligibility_heuristic_label
        - eligibility_ml_score
    score_threshold : float
        Threshold above which ML score is considered positive.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (false_positives, false_negatives)
    """
    _validate_inputs(df)

    df = df.copy()

    # ML binary prediction at threshold
    df["ml_predicted_positive"] = df["eligibility_ml_score"] >= score_threshold

    # False positives: ML positive, heuristic negative
    false_positives = df[
        (df["ml_predicted_positive"]) &
        (~df["eligibility_heuristic_label"])
    ]

    # False negatives: ML negative, heuristic positive
    false_negatives = df[
        (~df["ml_predicted_positive"]) &
        (df["eligibility_heuristic_label"])
    ]

    return false_positives, false_negatives


def summarise_errors(
    false_positives,
    false_negatives,
):
    """
    Summarise counts of false positives and false negatives.

    Parameters
    ----------
    false_positives : pd.DataFrame
        DataFrame of false positives.
    false_negatives : pd.DataFrame
        DataFrame of false negatives.

    Returns
    -------
    pd.DataFrame
        Summary table with counts.
    """
    summary = pd.DataFrame(
        {
            "error_type": ["false_positive", "false_negative"],
            "count": [len(false_positives), len(false_negatives)],
        }
    )

    return summary


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_inputs(df):
    required_columns = [
        "eligibility_heuristic_label",
        "eligibility_ml_score",
    ]

    require_columns(df, required_columns, context="Error analysis input")
    require_boolean(df, "eligibility_heuristic_label", context="Error analysis input")
    require_probability(df, "eligibility_ml_score", context="Error analysis input")
