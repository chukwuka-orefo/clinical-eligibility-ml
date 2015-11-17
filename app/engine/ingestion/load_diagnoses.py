# src/ingestion/load_diagnoses.py
"""
load_diagnoses.py

Load and normalise diagnosis data from MIMIC-III or MIMIC-IV.

Responsibilities:
- Read raw diagnosis tables
- Select and standardise required columns
- Preserve diagnosis ordering (primary vs secondary)
- Attach code-system metadata (ICD-9 vs ICD-10)
- Write an interim diagnoses table

This module must NOT:
- Apply clinical meaning to codes
- Define stroke or cardiovascular phenotypes
- Perform eligibility logic
"""

from pathlib import Path

import pandas as pd

from app.engine.config.paths import (
    MIMIC_III_DIR,
    MIMIC_IV_DIR,
    DIAGNOSES_INTERIM_PATH,
    ensure_directories,
)
from app.engine.config.settings import ACTIVE_DATASET, Dataset, ACTIVE_CODE_SYSTEM


# ---------------------------------------------------------------------
# Dataset-specific filenames
# ---------------------------------------------------------------------

MIMIC_III_DIAGNOSES_FILE = "DIAGNOSES_ICD.csv"
MIMIC_IV_DIAGNOSES_FILE = "diagnoses_icd.csv"


# ---------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------

MIMIC_III_COLUMNS = {
    "SUBJECT_ID": "subject_id",
    "HADM_ID": "hadm_id",
    "ICD9_CODE": "diagnosis_code",
    "SEQ_NUM": "seq_num",
}

MIMIC_IV_COLUMNS = {
    "subject_id": "subject_id",
    "hadm_id": "hadm_id",
    "icd_code": "diagnosis_code",
    "icd_version": "icd_version",
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_diagnoses(
    output_path=None,
):
    """
    Load diagnoses data for the active dataset and return a normalised DataFrame.

    Parameters
    ----------
    output_path : Optional[Path]
        If provided, write the interim diagnoses table to this path.
        Defaults to DIAGNOSES_INTERIM_PATH.

    Returns
    -------
    pd.DataFrame
        Normalised diagnoses table.
    """
    ensure_directories()

    if ACTIVE_DATASET == Dataset.MIMIC_III:
        df = _load_mimic_iii_diagnoses()
    elif ACTIVE_DATASET == Dataset.MIMIC_IV:
        df = _load_mimic_iv_diagnoses()
    else:
        raise ValueError("Unsupported dataset: {}".format(ACTIVE_DATASET))

    df = _postprocess_diagnoses(df)

    if output_path is None:
        output_path = DIAGNOSES_INTERIM_PATH

    df.to_csv(output_path, index=False)

    return df


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _load_mimic_iii_diagnoses():
    path = MIMIC_III_DIR / MIMIC_III_DIAGNOSES_FILE
    if not path.exists():
        raise FileNotFoundError("MIMIC-III diagnoses file not found at {}".format(path))

    df = pd.read_csv(
        path,
        usecols=list(MIMIC_III_COLUMNS.keys()),
    )

    df = df.rename(columns=MIMIC_III_COLUMNS)

    # Explicitly tag code system
    df["code_system"] = "ICD9"

    return df


def _load_mimic_iv_diagnoses():
    path = MIMIC_IV_DIR / MIMIC_IV_DIAGNOSES_FILE
    if not path.exists():
        raise FileNotFoundError("MIMIC-IV diagnoses file not found at {}".format(path))

    df = pd.read_csv(
        path,
        usecols=list(MIMIC_IV_COLUMNS.keys()),
    )

    df = df.rename(columns=MIMIC_IV_COLUMNS)

    # Map ICD version to code system
    df["code_system"] = df["icd_version"].map(
        {
            9: "ICD9",
            10: "ICD10",
        }
    )

    # Drop icd_version once mapped
    df = df.drop(columns=["icd_version"])

    return df


def _postprocess_diagnoses(df):
    """
    Perform minimal, non-clinical postprocessing:
    - enforce dtypes
    - normalise diagnosis codes
    - basic sanity checks
    """

    # Enforce integer IDs
    df["subject_id"] = df["subject_id"].astype(int)
    df["hadm_id"] = df["hadm_id"].astype(int)

    # Diagnosis codes as uppercase strings
    df["diagnosis_code"] = (
        df["diagnosis_code"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Sequence number may be missing in MIMIC-IV
    if "seq_num" in df.columns:
        df["seq_num"] = pd.to_numeric(df["seq_num"], errors="coerce")
    else:
        df["seq_num"] = None

    # Sanity checks
    if df["subject_id"].isnull().any():
        raise ValueError("Null subject_id detected in diagnoses data")

    if df["hadm_id"].isnull().any():
        raise ValueError("Null hadm_id detected in diagnoses data")

    if df["diagnosis_code"].isnull().any():
        raise ValueError("Null diagnosis_code detected in diagnoses data")

    if df["code_system"].isnull().any():
        raise ValueError("Unknown code_system detected in diagnoses data")

    # Ensure dataset-level consistency
    if ACTIVE_CODE_SYSTEM.value not in df["code_system"].unique():
        raise ValueError(
            "Active code system {} not found in diagnoses data".format(
                ACTIVE_CODE_SYSTEM.value
            )
        )

    return df
