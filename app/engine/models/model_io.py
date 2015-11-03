# app/engine/models/model_io.py

"""
model_io.py

Model persistence utilities for the Clinical Eligibility Engine.

Purpose:
- Save trained eligibility models to disk
- Load previously trained models
- Provide a single canonical location for model storage

This module intentionally keeps persistence simple and explicit.
It reflects common practice in 2014–2016 era applied ML projects.

This module MUST:
- Not perform training or inference
- Not contain business logic
- Not depend on Flask or UI code
"""

from pathlib import Path

import joblib


# ---------------------------------------------------------------------
# Canonical model location
# ---------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "trained_model.pkl"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def model_exists(model_path=DEFAULT_MODEL_PATH):
    """
    Check whether a trained model exists on disk.

    Parameters
    ----------
    model_path : Path
        Path to the model file.

    Returns
    -------
    bool
        True if model file exists, False otherwise.
    """
    return model_path.exists()


def save_model(
    model,
    model_path=DEFAULT_MODEL_PATH,
):
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : Any
        Trained model object.
    model_path : Path
        Destination path for the model file.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(
    model_path=DEFAULT_MODEL_PATH,
):
    """
    Load a trained model from disk.

    Parameters
    ----------
    model_path : Path
        Path to the model file.

    Returns
    -------
    Any
        Loaded model object.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found at {}".format(model_path)
        )

    return joblib.load(model_path)
