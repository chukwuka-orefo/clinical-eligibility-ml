# eligibility_label.py
"""
eligibility_label.py

Apply transparent, rule-based eligibility heuristics for stroke trial screening.

Responsibilities:
- Join admission context with stroke and cardiovascular phenotypes
- Apply inclusion and exclusion heuristics
- Produce a noisy baseline eligibility label

This module MUST:
- Operate at admission (hadm_id) level
- Be transparent and auditable
- Prioritise recall over precision

This module MUST NOT:
- Train models
- Perform feature scaling
- Encode categorical variables for ML
"""

from pathlib import Path

import pandas as pd

from app.engine.config.paths import (
    ADMISSIONS_INTERIM_PATH,
    INTERIM_DATA_DIR,
    ensure_directories,
)
from app.engine.config.thresholds import (
    MIN_ELIGIBLE_AGE,
    MAX_ELIGIBLE_AGE,
    MAX_EXCLUSION_AGE,
)
from app.engine.config.settings import INCLUDE_ELECTIVE_ADMISSIONS


# ---------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------

ELIGIBILITY_LABELS_PATH = INTERIM_DATA_DIR / "eligibility_heuristics.csv"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def derive_eligibility_labels(
    admissions_path=None,
    stroke_phenotype_path=None,
    cardiovascular_phenotype_path=None,
    output_path=None,
):
    """
    Derive heuristic eligibility labels for trial screening.

    Parameters
    ----------
    admissions_path : Optional[Path]
        Path to interim admissions table.
    stroke_phenotype_path : Optional[Path]
        Path to admission-level stroke phenotype table.
    cardiovascular_phenotype_path : Optional[Path]
        Path to admission-level cardiovascular phenotype table.
    output_path : Optional[Path]
        Path to write eligibility heuristic labels.

    Returns
    -------
    pd.DataFrame
        Admission-level eligibility heuristic table.
    """
    ensure_directories()

    admissions = _load_admissions(admissions_path)
    stroke = _load_stroke_phenotype(stroke_phenotype_path)
    cvd = _load_cardiovascular_phenotype(cardiovascular_phenotype_path)

    df = _join_inputs(admissions, stroke, cvd)
    df = _apply_heuristics(df)

    if output_path is None:
        output_path = ELIGIBILITY_LABELS_PATH

    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------

def _load_admissions(path):
    if path is None:
        path = ADMISSIONS_INTERIM_PATH
    if not path.exists():
        raise FileNotFoundError("Admissions file not found at {}".format(path))

    df = pd.read_csv(path)

    required = {
        "hadm_id",
        "subject_id",
        "admission_type",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError("Admissions table missing required columns: {}".format(missing))

    return df


def _load_stroke_phenotype(path):
    if path is None:
        path = INTERIM_DATA_DIR / "stroke_phenotype.csv"
    if not path.exists():
        raise FileNotFoundError("Stroke phenotype file not found at {}".format(path))

    df = pd.read_csv(path)

    required = {
        "hadm_id",
        "has_any_stroke_signal",
        "stroke_code_count",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError("Stroke phenotype table missing required columns: {}".format(missing))

    return df


def _load_cardiovascular_phenotype(path):
    if path is None:
        path = INTERIM_DATA_DIR / "cardiovascular_phenotype.csv"
    if not path.exists():
        raise FileNotFoundError("Cardiovascular phenotype file not found at {}".format(path))

    df = pd.read_csv(path)

    required = {
        "hadm_id",
        "has_any_cvd_signal",
        "cardiovascular_code_count",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError("Cardiovascular phenotype table missing required columns: {}".format(missing))

    return df


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

def _join_inputs(
    admissions,
    stroke,
    cvd,
):
    """
    Join admissions with phenotype tables at admission level.
    """

    df = admissions.merge(stroke, on="hadm_id", how="left")
    df = df.merge(cvd, on="hadm_id", how="left")

    # Fill missing phenotype signals conservatively
    df["has_any_stroke_signal"] = df["has_any_stroke_signal"].fillna(False)
    df["stroke_code_count"] = df["stroke_code_count"].fillna(0).astype(int)

    df["has_any_cvd_signal"] = df["has_any_cvd_signal"].fillna(False)
    df["cardiovascular_code_count"] = df["cardiovascular_code_count"].fillna(0).astype(int)

    return df


def _apply_heuristics(df):
    """
    Apply inclusion and exclusion heuristics to derive eligibility labels.
    """

    df = df.copy()

    # -----------------------------------------------------------------
    # Age-based rules (if age is available)
    # -----------------------------------------------------------------
    if "age_at_admission" in df.columns:
        df["age_in_range"] = (
            (df["age_at_admission"] >= MIN_ELIGIBLE_AGE) &
            (df["age_at_admission"] <= MAX_ELIGIBLE_AGE)
        )

        df["age_excluded"] = df["age_at_admission"] > MAX_EXCLUSION_AGE
    else:
        # If age is unavailable, do not exclude
        df["age_in_range"] = True
        df["age_excluded"] = False

    # -----------------------------------------------------------------
    # Stroke signal inclusion
    # -----------------------------------------------------------------
    df["stroke_signal_ok"] = df["has_any_stroke_signal"]

    # -----------------------------------------------------------------
    # Admission context
    # -----------------------------------------------------------------
    df["is_emergency"] = (
        df["admission_type"]
        .astype(str)
        .str.upper()
        .str.contains("EMERGENCY")
    )

    if INCLUDE_ELECTIVE_ADMISSIONS:
        df["admission_ok"] = True
    else:
        df["admission_ok"] = df["is_emergency"]

    # -----------------------------------------------------------------
    # Exclusion logic
    # -----------------------------------------------------------------
    df["excluded"] = df["age_excluded"]

    # -----------------------------------------------------------------
    # Final eligibility heuristic
    # -----------------------------------------------------------------
    df["eligibility_heuristic_label"] = (
        df["age_in_range"] &
        df["stroke_signal_ok"] &
        df["admission_ok"] &
        (~df["excluded"])
    )

    return df[
        [
            "subject_id",
            "hadm_id",
            "eligibility_heuristic_label",
            "has_any_stroke_signal",
            "stroke_code_count",
            "has_any_cvd_signal",
            "cardiovascular_code_count",
            "age_in_range",
            "admission_ok",
            "excluded",
        ]
    ]
