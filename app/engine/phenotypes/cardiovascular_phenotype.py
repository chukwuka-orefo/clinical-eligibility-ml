# src/phenotypes/cardiovascular_phenotype.py
"""
cardiovascular_phenotype.py

Derive admission-level cardiovascular phenotype features from diagnosis-level data.

Responsibilities:
- Aggregate diagnosis-level cardiovascular signals to admission level
- Compute interpretable cardiovascular phenotype features
- Provide contextual comorbidity signals for eligibility and ML

This module MUST:
- Operate at admission (hadm_id) granularity
- Treat diagnosis codes as signals, not ground truth
- Produce features usable for heuristics and ML

This module MUST NOT:
- Apply eligibility rules
- Perform feature scaling or encoding
- Train or score models
"""

from pathlib import Path

import pandas as pd

from app.engine.config.paths import (
    INTERIM_DATA_DIR,
    ensure_directories,
)


# ---------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------

CARDIOVASCULAR_PHENOTYPE_PATH = INTERIM_DATA_DIR / "cardiovascular_phenotype.csv"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def derive_cardiovascular_phenotype(
    diagnoses_with_codelists_path,
    output_path=None,
):
    """
    Derive cardiovascular phenotype features at admission level.

    Parameters
    ----------
    diagnoses_with_codelists_path : Path
        Path to diagnosis-level table annotated with codelist flags
        (output of apply_codelists.py).
    output_path : Optional[Path]
        Path to write admission-level cardiovascular phenotype table.
        Defaults to CARDIOVASCULAR_PHENOTYPE_PATH.

    Returns
    -------
    pd.DataFrame
        Admission-level cardiovascular phenotype features.
    """
    ensure_directories()

    if not diagnoses_with_codelists_path.exists():
        raise FileNotFoundError(
            "Annotated diagnoses file not found at {}".format(diagnoses_with_codelists_path)
        )

    df = pd.read_csv(diagnoses_with_codelists_path)

    _validate_input(df)

    phenotype_df = _aggregate_to_admission(df)

    if output_path is None:
        output_path = CARDIOVASCULAR_PHENOTYPE_PATH

    phenotype_df.to_csv(output_path, index=False)

    return phenotype_df


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _validate_input(df):
    required_columns = {
        "subject_id",
        "hadm_id",
        "diagnosis_code",
        "is_cardiovascular_code",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Diagnoses table missing required columns: {}".format(missing)
        )

    if not pd.api.types.is_bool_dtype(df["is_cardiovascular_code"]):
        raise TypeError("is_cardiovascular_code column must be boolean")

    if df["hadm_id"].isnull().any():
        raise ValueError("Null hadm_id detected in diagnoses table")


def _aggregate_to_admission(df):
    """
    Aggregate diagnosis-level cardiovascular signals to admission-level features.

    Produces:
    - cardiovascular_code_count
    - total_diagnosis_count
    - cardiovascular_code_density
    - has_any_cvd_signal
    """

    df = df.copy()

    # Count total diagnoses per admission
    total_dx = (
        df.groupby("hadm_id")
        .size()
        .rename("total_diagnosis_count")
    )

    # Count cardiovascular-related diagnoses per admission
    cvd_dx = (
        df[df["is_cardiovascular_code"]]
        .groupby("hadm_id")
        .size()
        .rename("cardiovascular_code_count")
    )

    phenotype = pd.concat([total_dx, cvd_dx], axis=1).fillna(0)

    # Convert counts to int
    phenotype["total_diagnosis_count"] = phenotype["total_diagnosis_count"].astype(int)
    phenotype["cardiovascular_code_count"] = phenotype["cardiovascular_code_count"].astype(int)

    # Cardiovascular density (contextual burden proxy)
    phenotype["cardiovascular_code_density"] = (
        phenotype["cardiovascular_code_count"]
        / phenotype["total_diagnosis_count"].replace(0, float("nan"))
    ).fillna(0.0)

    # Binary cardiovascular signal flag
    phenotype["has_any_cvd_signal"] = phenotype["cardiovascular_code_count"] > 0

    # Reset index to make hadm_id a column
    phenotype = phenotype.reset_index()

    return phenotype
