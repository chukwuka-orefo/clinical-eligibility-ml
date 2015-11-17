# src/code_mapping/apply_codelists.py

"""
apply_codelists.py

Apply clinical codelists to diagnosis-level data.

Responsibilities:
- Load interim diagnoses table
- Apply stroke and cardiovascular code mappings
- Annotate each diagnosis row with boolean flags
- Write an annotated interim table for downstream phenotype aggregation

This module MUST:
- Operate at diagnosis-row granularity
- Be deterministic and auditable
- Contain no aggregation logic

This module MUST NOT:
- Derive phenotypes
- Apply eligibility rules
- Perform feature engineering
"""

from pathlib import Path

import pandas as pd

from app.engine.config.paths import (
    DIAGNOSES_INTERIM_PATH,
    INTERIM_DATA_DIR,
    ensure_directories,
)
from code_mapping.stroke_codelist import is_stroke_code
from code_mapping.cardiovascular_codelist import is_cardiovascular_code


# ---------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------

ANNOTATED_DIAGNOSES_PATH = INTERIM_DATA_DIR / "diagnoses_with_codelists.csv"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def apply_codelists(
    diagnoses_path=None,
    output_path=None,
):
    """
    Apply stroke and cardiovascular codelists to diagnosis data.

    Parameters
    ----------
    diagnoses_path : Optional[Path]
        Path to interim diagnoses CSV.
        Defaults to DIAGNOSES_INTERIM_PATH.
    output_path : Optional[Path]
        Path to write annotated diagnoses table.
        Defaults to ANNOTATED_DIAGNOSES_PATH.

    Returns
    -------
    pd.DataFrame
        Diagnosis-level table with codelist flags applied.
    """
    ensure_directories()

    if diagnoses_path is None:
        diagnoses_path = DIAGNOSES_INTERIM_PATH

    if output_path is None:
        output_path = ANNOTATED_DIAGNOSES_PATH

    df = _load_diagnoses(diagnoses_path)
    df = _apply_codelists(df)

    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _load_diagnoses(path):
    if not path.exists():
        raise FileNotFoundError("Diagnoses file not found at {}".format(path))

    df = pd.read_csv(path)

    required_columns = {
        "subject_id",
        "hadm_id",
        "diagnosis_code",
        "code_system",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError("Diagnoses data missing required columns: {}".format(missing))

    return df


def _apply_codelists(df):
    """
    Apply stroke and cardiovascular codelists row-wise.

    Adds the following columns:
    - is_stroke_code
    - is_cardiovascular_code
    """

    df = df.copy()

    df["is_stroke_code"] = df.apply(
        lambda row: is_stroke_code(
            diagnosis_code=row["diagnosis_code"],
            code_system=row["code_system"],
        ),
        axis=1,
    )

    df["is_cardiovascular_code"] = df.apply(
        lambda row: is_cardiovascular_code(
            diagnosis_code=row["diagnosis_code"],
            code_system=row["code_system"],
        ),
        axis=1,
    )

    # Sanity check: flags must be boolean
    if not pd.api.types.is_bool_dtype(df["is_stroke_code"]):
        raise TypeError("is_stroke_code is not boolean")

    if not pd.api.types.is_bool_dtype(df["is_cardiovascular_code"]):
        raise TypeError("is_cardiovascular_code is not boolean")

    return df
