# src/phenotypes/stroke_phenotype.py
"""
stroke_phenotype.py

Derive admission-level stroke phenotype features from diagnosis-level data.

Responsibilities:
- Aggregate diagnosis-level stroke signals to admission level
- Compute interpretable stroke phenotype features
- Preserve uncertainty and signal strength (not binary diagnosis)

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

from config.paths import (
    INTERIM_DATA_DIR,
    ensure_directories,
)


# ---------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------

STROKE_PHENOTYPE_PATH = INTERIM_DATA_DIR / "stroke_phenotype.csv"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def derive_stroke_phenotype(
    diagnoses_with_codelists_path,
    output_path=None,
):
    """
    Derive stroke phenotype features at admission level.

    Parameters
    ----------
    diagnoses_with_codelists_path : Path
        Path to diagnosis-level table annotated with codelist flags
        (output of apply_codelists.py).
    output_path : Optional[Path]
        Path to write admission-level stroke phenotype table.
        Defaults to STROKE_PHENOTYPE_PATH.

    Returns
    -------
    pd.DataFrame
        Admission-level stroke phenotype features.
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
        output_path = STROKE_PHENOTYPE_PATH

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
        "is_stroke_code",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Diagnoses table missing required columns: {}".format(missing)
        )

    if not pd.api.types.is_bool_dtype(df["is_stroke_code"]):
        raise TypeError("is_stroke_code column must be boolean")

    if df["hadm_id"].isnull().any():
        raise ValueError("Null hadm_id detected in diagnoses table")


def _aggregate_to_admission(df):
    """
    Aggregate diagnosis-level stroke signals to admission-level features.

    Produces:
    - stroke_code_count
    - total_diagnosis_count
    - stroke_code_density
    - has_any_stroke_signal
    - stroke_primary_dx_flag (if seq_num available)
    """

    df = df.copy()

    # Count total diagnoses per admission
    total_dx = (
        df.groupby("hadm_id")
        .size()
        .rename("total_diagnosis_count")
    )

    # Count stroke-related diagnoses per admission
    stroke_dx = (
        df[df["is_stroke_code"]]
        .groupby("hadm_id")
        .size()
        .rename("stroke_code_count")
    )

    phenotype = pd.concat([total_dx, stroke_dx], axis=1).fillna(0)

    # Convert counts to int
    phenotype["total_diagnosis_count"] = phenotype["total_diagnosis_count"].astype(int)
    phenotype["stroke_code_count"] = phenotype["stroke_code_count"].astype(int)

    # Stroke density (signal strength proxy)
    phenotype["stroke_code_density"] = (
        phenotype["stroke_code_count"]
        / phenotype["total_diagnosis_count"].replace(0, pd.NA)
    ).fillna(0.0)

    # Binary stroke signal flag
    phenotype["has_any_stroke_signal"] = phenotype["stroke_code_count"] > 0

    # Optional: primary diagnosis stroke flag (only if seq_num exists)
    if "seq_num" in df.columns:
        primary_stroke = (
            df[
                (df["seq_num"] == 1) &
                (df["is_stroke_code"])
            ]
            .groupby("hadm_id")
            .size()
            .rename("stroke_primary_dx_flag")
        )

        phenotype = phenotype.join(primary_stroke, how="left")
        phenotype["stroke_primary_dx_flag"] = (
            phenotype["stroke_primary_dx_flag"]
            .fillna(0)
            .astype(int)
            .astype(bool)
        )
    else:
        phenotype["stroke_primary_dx_flag"] = False

    # Reset index to make hadm_id a column
    phenotype = phenotype.reset_index()

    return phenotype
