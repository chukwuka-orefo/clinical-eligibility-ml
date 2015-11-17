# src/features/build_feature_matrix.py
"""
build_feature_matrix.py

Build the final modelling-ready feature matrix for Trial Eligibility ML.

Responsibilities:
- Join admissions, phenotype, and heuristic outputs
- Select and validate ML input features
- Enforce admission-level grain
- Write processed feature matrix

This module MUST:
- Produce exactly one row per hadm_id
- Contain no model logic
- Contain no feature scaling or encoding
- Prevent leakage by construction
"""

from pathlib import Path

import pandas as pd

from app.engine.config.paths import (
    ADMISSIONS_INTERIM_PATH,
    PROCESSED_FEATURES_PATH,
    INTERIM_DATA_DIR,
    ensure_directories,
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_feature_matrix(
    admissions_path=None,
    stroke_phenotype_path=None,
    cardiovascular_phenotype_path=None,
    eligibility_heuristics_path=None,
    output_path=None,
):
    """
    Build the final modelling feature matrix.

    Parameters
    ----------
    admissions_path : Optional[Path]
        Path to admissions interim table.
    stroke_phenotype_path : Optional[Path]
        Path to stroke phenotype table.
    cardiovascular_phenotype_path : Optional[Path]
        Path to cardiovascular phenotype table.
    eligibility_heuristics_path : Optional[Path]
        Path to eligibility heuristic table.
    output_path : Optional[Path]
        Path to write processed feature matrix.

    Returns
    -------
    pd.DataFrame
        Final modelling-ready feature table.
    """
    ensure_directories()

    admissions = _load_admissions(admissions_path)
    stroke = _load_stroke_phenotype(stroke_phenotype_path)
    cvd = _load_cardiovascular_phenotype(cardiovascular_phenotype_path)
    heuristics = _load_eligibility_heuristics(eligibility_heuristics_path)

    df = _join_all(admissions, stroke, cvd, heuristics)
    df = _validate_feature_matrix(df)

    if output_path is None:
        output_path = PROCESSED_FEATURES_PATH

    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def _load_admissions(path):
    if path is None:
        path = ADMISSIONS_INTERIM_PATH
    if not path.exists():
        raise FileNotFoundError("Admissions file not found at {}".format(path))
    return pd.read_csv(path)


def _load_stroke_phenotype(path):
    if path is None:
        path = INTERIM_DATA_DIR / "stroke_phenotype.csv"
    if not path.exists():
        raise FileNotFoundError("Stroke phenotype file not found at {}".format(path))
    return pd.read_csv(path)


def _load_cardiovascular_phenotype(path):
    if path is None:
        path = INTERIM_DATA_DIR / "cardiovascular_phenotype.csv"
    if not path.exists():
        raise FileNotFoundError("Cardiovascular phenotype file not found at {}".format(path))
    return pd.read_csv(path)


def _load_eligibility_heuristics(path):
    if path is None:
        path = INTERIM_DATA_DIR / "eligibility_heuristics.csv"
    if not path.exists():
        raise FileNotFoundError("Eligibility heuristics file not found at {}".format(path))
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

def _join_all(
    admissions,
    stroke,
    cvd,
    heuristics,
):
    """
    Join all pipeline outputs into a single admission-level table.
    """

    df = admissions.merge(stroke, on="hadm_id", how="left")
    df = df.merge(cvd, on="hadm_id", how="left")
    df = df.merge(
        heuristics[
            [
                "hadm_id",
                "eligibility_heuristic_label",
            ]
        ],
        on="hadm_id",
        how="left",
    )

    # Conservative defaults for missing phenotype data
    phenotype_defaults = {
        "stroke_code_count": 0,
        "stroke_code_density": 0.0,
        "has_any_stroke_signal": False,
        "stroke_primary_dx_flag": False,
        "cardiovascular_code_count": 0,
        "cardiovascular_code_density": 0.0,
        "has_any_cvd_signal": False,
    }

    for col, default in phenotype_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # Eligibility label defaults to False if missing
    df["eligibility_heuristic_label"] = (
        df["eligibility_heuristic_label"]
        .fillna(False)
        .astype(bool)
    )

    return df


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_feature_matrix(df):
    """
    Validate that the feature matrix meets modelling requirements.
    """

    required_columns = {
        "subject_id",
        "hadm_id",
        "eligibility_heuristic_label",
        "stroke_code_count",
        "stroke_code_density",
        "has_any_stroke_signal",
        "cardiovascular_code_count",
        "cardiovascular_code_density",
        "has_any_cvd_signal",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError("Feature matrix missing required columns: {}".format(missing))

    # Ensure one row per admission
    if df.duplicated(subset=["hadm_id"]).any():
        raise ValueError("Duplicate hadm_id detected in feature matrix")

    # Ensure boolean columns are boolean
    bool_columns = [
        "eligibility_heuristic_label",
        "has_any_stroke_signal",
        "has_any_cvd_signal",
    ]

    for col in bool_columns:
        if not pd.api.types.is_bool_dtype(df[col]):
            raise TypeError("Column {} must be boolean".format(col))

    return df
