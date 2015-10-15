# src/models/base.py
"""
base.py

Abstract base class and shared helpers for Trial Eligibility ML models.

Purpose:
- Define a minimal, explicit model interface
- Enforce consistency across baseline and future models
- Keep orchestration code simple and generic

This module MUST:
- Define common method signatures
- Avoid framework-specific assumptions
- Remain lightweight and dependency-free

This module MUST NOT:
- Perform data loading
- Perform feature engineering
- Encode model-specific logic
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseEligibilityModel(ABC):
    """
    Abstract base class for eligibility models.

    All models must:
    - be trainable on a feature matrix
    - produce a continuous eligibility score
    - operate at admission-level granularity
    """

    @abstractmethod
    def fit(
        self,
        X,
        y,
    ):
        """
        Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target labels (heuristic proxy).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_score(
        self,
        X,
    ):
        """
        Predict eligibility scores for ranking.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        pd.Series
            Continuous eligibility scores indexed to X.
        """
        raise NotImplementedError

    @abstractmethod
    def get_feature_names(self):
        """
        Return the list of feature names used by the model.

        Returns
        -------
        Iterable[str]
            Feature names.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------
# Shared helpers (optional)
# ---------------------------------------------------------------------

def validate_feature_matrix(
    X,
    feature_names,
):
    """
    Validate that required feature columns exist in X.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    feature_names : Iterable[str]
        Expected feature names.

    Raises
    ------
    ValueError
        If required features are missing.
    """
    missing = set(feature_names) - set(X.columns)
    if missing:
        raise ValueError(
            "Feature matrix missing required features: {}".format(sorted(missing))
        )


def ensure_numeric_features(
    X,
    feature_names,
):
    """
    Ensure that specified features are numeric.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    feature_names : Iterable[str]
        Features expected to be numeric.

    Raises
    ------
    TypeError
        If any feature is non-numeric.
    """
    for name in feature_names:
        if not pd.api.types.is_numeric_dtype(X[name]):
            raise TypeError(
                "Feature '{}' must be numeric".format(name)
            )
