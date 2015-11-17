# src/evaluation/compare_to_heuristic.py
"""
compare_to_heuristic.py

Comparison utilities for heuristic screening vs ML-based ranking.

Purpose:
- Provide a single, interpretable comparison of screening strategies
- Summarise recall and precision at fixed screening capacities
- Support reporting and UI display

This module MUST:
- Operate on scored admission-level data
- Treat heuristic labels as noisy reference
- Produce simple, tabular outputs

This module MUST NOT:
- Train models
- Modify scores or labels
- Perform plotting or visualisation
"""

import pandas as pd

from evaluation.ranking import evaluate_ranking
from app.engine.config.thresholds import DEFAULT_SCREENING_K_VALUES
from app.engine.utils.checks import (
    require_columns,
    require_boolean,
    require_probability,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def compare_screening_strategies(
    df,
    k_values=None,
):
    """
    Compare heuristic-only screening to ML-based ranking.

    Parameters
    ----------
    df : pd.DataFrame
        Scored admissions table containing:
        - eligibility_heuristic_label
        - eligibility_ml_score
    k_values : Iterable[int], optional
        Screening capacities to evaluate.
        Defaults to DEFAULT_SCREENING_K_VALUES.

    Returns
    -------
    pd.DataFrame
        Comparison table with recall and precision metrics.
    """
    if k_values is None:
        k_values = DEFAULT_SCREENING_K_VALUES

    _validate_inputs(df)

    results = evaluate_ranking(df, k_values)

    # Pivot to make comparison explicit
    comparison = (
        results
        .pivot_table(
            index="k",
            columns="method",
            values=["recall_at_k", "precision_at_k"],
        )
        .reset_index()
    )

    # Flatten multi-level columns
    comparison.columns = [
        "_".join(col).strip("_")
        for col in comparison.columns.values
    ]

    return comparison


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_inputs(df):
    required_columns = [
        "eligibility_heuristic_label",
        "eligibility_ml_score",
    ]

    require_columns(df, required_columns, context="Heuristic comparison input")
    require_boolean(df, "eligibility_heuristic_label", context="Heuristic comparison input")
    require_probability(df, "eligibility_ml_score", context="Heuristic comparison input")
