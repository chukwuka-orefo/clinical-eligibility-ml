# src/evaluation/ranking.py
"""
ranking.py

Ranking-based evaluation for Trial Eligibility ML.

Responsibilities:
- Evaluate screening efficiency using ranking metrics
- Compare heuristic-only screening vs ML-ranked screening
- Compute Recall@K and Precision@K
- Support multiple K thresholds

This module MUST:
- Operate on scored admission-level data
- Treat eligibility heuristics as noisy ground truth
- Focus on ranking quality, not classification accuracy

This module MUST NOT:
- Train models
- Modify feature values
"""

import pandas as pd

from app.engine.config.settings import LOG_LEVEL
from app.engine.utils.logging import get_logger


LOGGER = get_logger(__name__, LOG_LEVEL)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def evaluate_ranking(
    df,
    k_values,
):
    """
    Evaluate ranking performance for heuristic vs ML screening.

    Parameters
    ----------
    df : pd.DataFrame
        Scored admissions table containing:
        - eligibility_heuristic_label
        - eligibility_ml_score
    k_values : Iterable[int]
        Values of K to evaluate (e.g. [50, 100, 200])

    Returns
    -------
    pd.DataFrame
        Ranking metrics per method and K.
    """
    _validate_inputs(df)

    results = []

    for k in k_values:
        LOGGER.info("Evaluating ranking metrics at K={}".format(k))

        # Heuristic-only screening (binary, unordered)
        heuristic_metrics = _evaluate_heuristic_screening(df, k)
        heuristic_metrics["method"] = "heuristic"
        heuristic_metrics["k"] = k
        results.append(heuristic_metrics)

        # ML-ranked screening
        ml_metrics = _evaluate_ml_ranking(df, k)
        ml_metrics["method"] = "ml"
        ml_metrics["k"] = k
        results.append(ml_metrics)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Internal evaluation methods
# ---------------------------------------------------------------------

def _evaluate_heuristic_screening(
    df,
    k,
):
    """
    Evaluate screening by heuristic label only.

    This simulates a naive workflow where:
    - All heuristic-positive cases are reviewed
    - No prioritisation is applied
    """
    positives = df[df["eligibility_heuristic_label"]]

    screened = positives.head(k)
    total_positives = positives.shape[0]

    recall = (
        screened.shape[0] / total_positives
        if total_positives > 0 else 0.0
    )

    precision = (
        screened.shape[0] / k
        if k > 0 else 0.0
    )

    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
    }


def _evaluate_ml_ranking(
    df,
    k,
):
    """
    Evaluate ML-based ranking.

    Admissions are ranked by ML eligibility score.
    """
    ranked = df.sort_values(
        by="eligibility_ml_score",
        ascending=False,
    )

    screened = ranked.head(k)

    total_positives = df["eligibility_heuristic_label"].sum()
    true_positives = screened["eligibility_heuristic_label"].sum()

    recall = (
        true_positives / total_positives
        if total_positives > 0 else 0.0
    )

    precision = (
        true_positives / k
        if k > 0 else 0.0
    )

    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_inputs(df):
    required_columns = {
        "eligibility_heuristic_label",
        "eligibility_ml_score",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Ranking evaluation missing required columns: {}".format(missing)
        )

    if not pd.api.types.is_bool_dtype(df["eligibility_heuristic_label"]):
        raise TypeError(
            "eligibility_heuristic_label must be boolean"
        )

    if df["eligibility_ml_score"].isnull().any():
        raise ValueError(
            "Null values detected in eligibility_ml_score"
        )
