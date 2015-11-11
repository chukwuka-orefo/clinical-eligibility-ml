# app/engine/models/model_manager.py
"""
model_manager.py

Model lifecycle management for the Clinical Eligibility Engine.

Purpose:
- Decide whether to train a new model or load an existing one
- Coordinate persistence of trained models
- Keep orchestration logic out of run_engine.py

This module intentionally does NOT:
- perform feature engineering
- define model architectures
- expose UI concerns

It represents the boundary between analytical workflows
(training) and operational workflows (screening).
"""

from pathlib import Path

import pandas as pd

from engine.models.model_io import (
    model_exists,
    save_model,
    load_model,
)
from engine.models.train import train_models


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def get_or_train_model(
    model_path,
    force_retrain=False,
):
    """
    Load an existing trained model or train a new one if required.

    Parameters
    ----------
    model_path : Path
        Path to the persisted model file.
    force_retrain : bool
        If True, retrain the model even if one already exists.

    Returns
    -------
    Tuple[pd.DataFrame, object, str]
        - scored dataframe
        - trained model object
        - mode ("trained" or "loaded")
    """

    if model_exists(model_path) and not force_retrain:
        model = load_model(model_path)
        scored_df = train_models()
        mode = "loaded"
        return scored_df, model, mode

    # Train a new model using the existing pipeline
    scored_df, model = train_models(return_model=True)
    save_model(model, model_path)
    mode = "trained"

    return scored_df, model, mode
