# src/phenotypes/stroke_phenotype.py
"""
stroke_phenotype.py

Derive admission-level stroke phenotype features from diagnosis-level data.
"""

import pandas as pd

from app.engine.config.paths import (
    INTERIM_DATA_DIR,
    ensure_directories,
)


# ---------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------

STROKE_PHENOTYPE_PATH = INTERIM_DATA_DIR / "stroke_phenotype.csv"

STROKE_PREFIXES = ("I61", "I63", "I64")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def derive_stroke_phenotype(
    diagnoses_with_codelists_path,
    output_path=None,
):
    """
    Derive stroke phenotype features at admission level.
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

    # Derive stroke code flag if not already present
    if "is_stroke_code" not in df.columns:
        df = df.copy()
        df["is_stroke_code"] = (
            (df["code_system"] == "ICD10") &
            df["diagnosis_code"].astype(str).str.startswith(STROKE_PREFIXES)
        )

    phenotype_df = _aggregate_to_admission(df)

    if output_path is None:
        output_path = STROKE_PHENOTYPE_PATH

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
    Aggregate diagnosis-level stroke signals to admission-level features.
    """
    total_dx = (
        df.groupby("hadm_id")
        .size()
        .rename("total_diagnosis_count")
    )

    stroke_dx = (
        df[df["is_stroke_code"]]
        .groupby("hadm_id")
        .size()
        .rename("stroke_code_count")
    )

    phenotype = pd.concat([total_dx, stroke_dx], axis=1).fillna(0)

    phenotype["total_diagnosis_count"] = (
        phenotype["total_diagnosis_count"].astype(int)
    )
    phenotype["stroke_code_count"] = (
        phenotype["stroke_code_count"].astype(int)
    )

    phenotype["stroke_code_density"] = (
        phenotype["stroke_code_count"] /
        phenotype["total_diagnosis_count"].replace(0, float("nan"))
    ).fillna(0.0)

    phenotype["has_any_stroke_signal"] = (
        phenotype["stroke_code_count"] > 0
    )

    if "seq_num" in df.columns:
        primary = (
            df[
                (df["seq_num"] == 1) &
                (df["is_stroke_code"])
            ]
            .groupby("hadm_id")
            .size()
            .rename("stroke_primary_dx_flag")
        )

        phenotype = phenotype.join(primary, how="left")
        phenotype["stroke_primary_dx_flag"] = (
            phenotype["stroke_primary_dx_flag"]
            .fillna(0)
            .astype(int)
            .astype(bool)
        )
    else:
        phenotype["stroke_primary_dx_flag"] = False

    return phenotype.reset_index()
