# src/evaluation/metrics.py
"""
metrics.py

Auxiliary evaluation metrics for Trial Eligibility ML.

Responsibilities:
- Compute standard classification metrics for sanity checking
- Assess calibration of ML scores
- Support comparison across models

These metrics are SECONDARY to ranking metrics and are used to:
- detect gross model failures
- compare models consistently
- provide quantitative context

This module MUST:
- Treat heuristic labels as noisy proxies
- Avoid overinterpreting absolute metric values

This module MUST NOT:
- Drive model selection in isolation
- Be used to claim clinical validity
"""

import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from config.settings import LOG_LEVEL
from utils.logging import get_logger


LOGGER = get_logger(__name__, LOG_LEVEL)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def compute_metrics(df):
    """
    Compute auxiliary evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Scored admissions table containing:
        - eligibility_heuristic_label (bool)
        - eligibility_ml_score (float)

    Returns
    -------
    Dict[str, float]
        Dictionary of computed metrics.
    """
    _validate_inputs(df)

    y_true = df["eligibility_heuristic_label"].astype(int)
    y_score = df["eligibility_ml_score"].astype(float)

    metrics = {}

    # ROC-AUC (sanity check, not primary objective)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        LOGGER.warning("ROC-AUC could not be computed (single-class labels)")
        metrics["roc_auc"] = np.nan

    # PR-AUC (useful under class imbalance)
    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    except ValueError:
        LOGGER.warning("PR-AUC could not be computed (single-class labels)")
        metrics["pr_auc"] = np.nan

    # Brier score (calibration quality)
    try:
        metrics["brier_score"] = brier_score_loss(y_true, y_score)
    except ValueError:
        LOGGER.warning("Brier score could not be computed")
        metrics["brier_score"] = np.nan

    # Prevalence for context
    metrics["positive_rate"] = float(y_true.mean())

    return metrics


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
            "Metrics evaluation missing required columns: {}".format(missing)
        )

    if not pd.api.types.is_bool_dtype(df["eligibility_heuristic_label"]):
        raise TypeError(
            "eligibility_heuristic_label must be boolean"
        )

    if df["eligibility_ml_score"].isnull().any():
        raise ValueError(
            "Null values detected in eligibility_ml_score"
        )

    if not np.isfinite(df["eligibility_ml_score"]).all():
        raise ValueError(
            "Non-finite values detected in eligibility_ml_score"
        )
