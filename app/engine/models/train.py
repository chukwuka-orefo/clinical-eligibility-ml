# src/models/train.py
"""
train.py

Training orchestration for Trial Eligibility ML models.

Responsibilities:
- Load processed feature matrix
- Train baseline ML models
- Attach ML scores to admissions
- Write scored outputs for evaluation

This module MUST:
- Contain no feature engineering logic
- Contain no evaluation logic
- Be deterministic and reproducible

This module MUST NOT:
- Perform train/test splitting (evaluation handles this)
- Optimise hyperparameters
"""

from pathlib import Path

import pandas as pd

from app.engine.config.paths import (
    PROCESSED_FEATURES_PATH,
    PROCESSED_DATA_DIR,
    ensure_directories,
)
from app.engine.models.logistic_regression import (
    train_logistic_regression,
)
from app.engine.config.settings import LOG_LEVEL
from app.engine.utils.logging import get_logger


LOGGER = get_logger(__name__, LOG_LEVEL)

# ---------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------

SCORED_OUTPUT_PATH = PROCESSED_DATA_DIR / "trial_eligibility_scored.csv"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def train_models(
    feature_matrix_path=None,
    output_path=None,
):
    """
    Train baseline models and attach ML scores.

    Parameters
    ----------
    feature_matrix_path : Optional[Path]
        Path to processed feature matrix.
        Defaults to PROCESSED_FEATURES_PATH.
    output_path : Optional[Path]
        Path to write scored output table.

    Returns
    -------
    pd.DataFrame
        Feature matrix with ML eligibility scores attached.
    """
    ensure_directories()

    if feature_matrix_path is None:
        feature_matrix_path = PROCESSED_FEATURES_PATH

    if not feature_matrix_path.exists():
        raise FileNotFoundError(
            "Processed feature matrix not found at {}".format(feature_matrix_path)
        )

    LOGGER.info("Loading processed feature matrix")
    df = pd.read_csv(feature_matrix_path)

    LOGGER.info("Training logistic regression baseline")
    model, scores = train_logistic_regression(df)

    LOGGER.info("Attaching ML scores to feature matrix")
    df = df.copy()
    df["eligibility_ml_score"] = scores.values

    if output_path is None:
        output_path = SCORED_OUTPUT_PATH

    LOGGER.info("Writing scored output to {}".format(output_path))
    df.to_csv(output_path, index=False)

    return df


# ---------------------------------------------------------------------
# CLI entry point (optional)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    train_models()
