# src/phenotypes/cardiovascular_phenotype.py
"""
cardiovascular_phenotype.py

Derive admission-level cardiovascular phenotype features from diagnosis-level data.
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

CARDIOVASCULAR_PHENOTYPE_PATH = (
    INTERIM_DATA_DIR / "cardiovascular_phenotype.csv"
)

CARDIOVASCULAR_PREFIXES = (
    "I20", "I21", "I22", "I23", "I24", "I25",
    "I50",
    "I60", "I61", "I62", "I63", "I64",
)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def derive_cardiovascular_phenotype(
    diagnoses_with_codelists_path,
    output_path=None,
):
    """
    Derive cardiovascular phenotype features at admission level.
    """
    ensure_directories()

    if not diagnoses_with_codelists_path.exists():
        raise FileNotFoundError(
            "Annotated diagnoses file not found at {0}".format(
                diagnoses_with_codelists_path
            )
        )

    df = pd.read_csv(diagnoses_with_codelists_path)
    _validate_input(df)

    # Derive cardiovascular code flag if missing
    if "is_cardiovascular_code" not in df.columns:
        df = df.copy()
        df["is_cardiovascular_code"] = (
            (df["code_system"] == "ICD10") &
            df["diagnosis_code"].astype(str).str.startswith(
                CARDIOVASCULAR_PREFIXES
            )
        )

    phenotype_df = _aggregate_to_admission(df)

    if output_path is None:
        output_path = CARDIOVASCULAR_PHENOTYPE_PATH

    phenotype_df.to_csv(output_path, index=False)
    return phenotype_df


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _validate_input(df):
    required_columns = set([
        "subject_id",
        "hadm_id",
        "diagnosis_code",
        "code_system",
    ])

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Diagnoses table missing required columns: {0}".format(missing)
        )

    if df["hadm_id"].isnull().any():
        raise ValueError("Null hadm_id detected in diagnoses table")


def _aggregate_to_admission(df):
    """
    Aggregate diagnosis-level cardiovascular signals to admission-level features.
    """
    total_dx = (
        df.groupby("hadm_id")
        .size()
        .rename("total_diagnosis_count")
    )

    cardio_dx = (
        df[df["is_cardiovascular_code"]]
        .groupby("hadm_id")
        .size()
        .rename("cardiovascular_code_count")
    )

    phenotype = pd.concat([total_dx, cardio_dx], axis=1).fillna(0)

    phenotype["total_diagnosis_count"] = (
        phenotype["total_diagnosis_count"].astype(int)
    )
    phenotype["cardiovascular_code_count"] = (
        phenotype["cardiovascular_code_count"].astype(int)
    )

    phenotype["cardiovascular_code_density"] = (
        phenotype["cardiovascular_code_count"] /
        phenotype["total_diagnosis_count"].replace(0, float("nan"))
    ).fillna(0.0)

    phenotype["has_any_cardiovascular_signal"] = (
        phenotype["cardiovascular_code_count"] > 0
    )

    return phenotype.reset_index()
