# src/models/decision_tree.py
"""
decision_tree.py

Decision tree baseline model for Trial Eligibility ML.

Purpose:
- Explain an interpretable, non-linear baseline
- Mirror heuristic-style decision logic
- Act as a contrast to logistic regression

This model is intentionally simple and constrained to avoid overfitting.
"""

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from models.base import (
    BaseEligibilityModel,
    validate_feature_matrix,
    ensure_numeric_features,
)
from models.logistic_regression import DEFAULT_FEATURE_COLUMNS
from app.engine.config.settings import LOG_LEVEL
from app.engine.utils.logging import get_logger


LOGGER = get_logger(__name__, LOG_LEVEL)


class DecisionTreeEligibilityModel(BaseEligibilityModel):
    """
    Shallow decision tree for eligibility ranking.

    Designed to:
    - Capture simple interactions
    - Remain interpretable
    - Avoid deep, unstable trees
    """

    def __init__(
        self,
        feature_names=None,
        max_depth=3,
        min_samples_leaf=50,
        random_state=42,
    ):
        self.feature_names = list(feature_names) if feature_names else list(DEFAULT_FEATURE_COLUMNS)

        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
        )

    # -----------------------------------------------------------------
    # BaseEligibilityModel interface
    # -----------------------------------------------------------------

    def fit(
        self,
        X,
        y,
    ):
        """
        Fit the decision tree model.
        """
        validate_feature_matrix(X, self.feature_names)
        ensure_numeric_features(X, self.feature_names)

        LOGGER.info("Training decision tree eligibility model")
        self.model.fit(X[self.feature_names], y.astype(int))

    def predict_score(
        self,
        X,
    ):
        """
        Predict eligibility scores for ranking.

        Uses class probability for the positive (eligible) class.
        """
        validate_feature_matrix(X, self.feature_names)
        ensure_numeric_features(X, self.feature_names)

        probs = self.model.predict_proba(X[self.feature_names])[:, 1]

        return pd.Series(
            probs,
            index=X.index,
            name="eligibility_ml_score",
        )

    def get_feature_names(self):
        """
        Return feature names used by the model.
        """
        return self.feature_names
