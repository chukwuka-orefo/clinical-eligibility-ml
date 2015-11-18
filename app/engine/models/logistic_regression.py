# src/models/logistic_regression.py
"""
logistic_regression.py

Baseline logistic regression model for trial eligibility inference.

Responsibilities:
- Train a logistic regression model on processed feature matrix
- Produce eligibility scores (probabilities) for ranking
- Provide transparent, interpretable coefficients

This model is intended as:
- The primary ML baseline
- A refinement over heuristic screening
- A reference point for more complex models

This module MUST NOT:
- Perform feature scaling implicitly
- Perform train/test splitting (handled elsewhere)
- Encode categorical variables automatically
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from app.engine.config.settings import LOG_LEVEL
from app.engine.utils.logging import get_logger


LOGGER = get_logger(__name__, LOG_LEVEL)


# ---------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------

DEFAULT_FEATURE_COLUMNS = [
    "stroke_code_count",
    "stroke_code_density",
    "has_any_stroke_signal",
    "cardiovascular_code_count",
    "cardiovascular_code_density",
    "has_any_cardiovascular_signal",
]


TARGET_COLUMN = "eligibility_heuristic_label"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def train_logistic_regression(
    df,
    feature_columns=None,
):
    """
    Train a logistic regression model and return predicted probabilities.

    Parameters
    ----------
    df : pd.DataFrame
        Processed feature matrix.
    feature_columns : List[str], optional
        Columns to use as model features.
        Defaults to DEFAULT_FEATURE_COLUMNS.

    Returns
    -------
    model : LogisticRegression
        Trained logistic regression model.
    scores : pd.Series
        Predicted eligibility probabilities indexed by hadm_id.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS

    _validate_inputs(df, feature_columns)

    X = df[feature_columns].astype(float)
    y = df[TARGET_COLUMN].astype(int)

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    LOGGER.info("Training logistic regression model")
    model.fit(X, y)

    LOGGER.info("Scoring admissions with logistic regression")
    scores = pd.Series(
        model.predict_proba(X)[:, 1],
        index=df["hadm_id"],
        name="eligibility_ml_score",
    )

    return model, scores


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def get_model_coefficients(
    model,
    feature_columns=None,
):
    """
    Return model coefficients in a readable format.

    Parameters
    ----------
    model : LogisticRegression
        Trained model.
    feature_columns : List[str], optional
        Feature names corresponding to coefficients.

    Returns
    -------
    pd.DataFrame
        Coefficients sorted by absolute magnitude.
    """
    if feature_columns is None:
        feature_columns = DEFAULT_FEATURE_COLUMNS

    coefs = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": model.coef_[0],
        }
    )

    coefs["abs_coefficient"] = np.abs(coefs["coefficient"])
    coefs = coefs.sort_values("abs_coefficient", ascending=False)

    return coefs.reset_index(drop=True)


def _validate_inputs(df, feature_columns):
    required_columns = set(feature_columns + [TARGET_COLUMN, "hadm_id"])

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError("Missing required columns for modelling: {}".format(missing))

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError("Null values detected in target column")

    for col in feature_columns:
        if df[col].isnull().any():
            raise ValueError("Null values detected in feature column: {}".format(col))
