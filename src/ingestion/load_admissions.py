# src/ingestion/load_admissions.py
"""
load_admissions.py

Load and normalise hospital admissions data from MIMIC-III or MIMIC-IV.

Responsibilities:
- Read raw admissions tables
- Select and standardise required columns
- Derive basic, non-leaky fields (e.g. length of stay)
- Perform minimal sanity checks
- Write an interim admissions table

This module must NOT:
- Apply phenotype logic
- Apply eligibility rules
- Perform feature engineering
"""

from pathlib import Path

import pandas as pd

from config.paths import (
    MIMIC_III_DIR,
    MIMIC_IV_DIR,
    ADMISSIONS_INTERIM_PATH,
    ensure_directories,
)
from config.settings import ACTIVE_DATASET, Dataset


# ---------------------------------------------------------------------
# Dataset-specific filenames
# ---------------------------------------------------------------------

MIMIC_III_ADMISSIONS_FILE = "ADMISSIONS.csv"
MIMIC_IV_ADMISSIONS_FILE = "admissions.csv"


# ---------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------

MIMIC_III_COLUMNS = {
    "SUBJECT_ID": "subject_id",
    "HADM_ID": "hadm_id",
    "ADMITTIME": "admittime",
    "DISCHTIME": "dischtime",
    "ADMISSION_TYPE": "admission_type",
    "ETHNICITY": "ethnicity",
}

MIMIC_IV_COLUMNS = {
    "subject_id": "subject_id",
    "hadm_id": "hadm_id",
    "admittime": "admittime",
    "dischtime": "dischtime",
    "admission_type": "admission_type",
    "ethnicity": "ethnicity",
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_admissions(
    output_path=None,
):
    """
    Load admissions data for the active dataset and return a normalised DataFrame.

    Parameters
    ----------
    output_path : Optional[Path]
        If provided, write the interim admissions table to this path.
        Defaults to ADMISSIONS_INTERIM_PATH.

    Returns
    -------
    pd.DataFrame
        Normalised admissions table.
    """
    ensure_directories()

    if ACTIVE_DATASET == Dataset.MIMIC_III:
        df = _load_mimic_iii_admissions()
    elif ACTIVE_DATASET == Dataset.MIMIC_IV:
        df = _load_mimic_iv_admissions()
    else:
        raise ValueError("Unsupported dataset: {}".format(ACTIVE_DATASET))

    df = _postprocess_admissions(df)

    if output_path is None:
        output_path = ADMISSIONS_INTERIM_PATH

    df.to_csv(output_path, index=False)

    return df


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _load_mimic_iii_admissions():
    path = MIMIC_III_DIR / MIMIC_III_ADMISSIONS_FILE
    if not path.exists():
        raise FileNotFoundError("MIMIC-III admissions file not found at {}".format(path))

    df = pd.read_csv(
        path,
        usecols=list(MIMIC_III_COLUMNS.keys()),
        parse_dates=["ADMITTIME", "DISCHTIME"],
    )

    df = df.rename(columns=MIMIC_III_COLUMNS)
    return df


def _load_mimic_iv_admissions():
    path = MIMIC_IV_DIR / MIMIC_IV_ADMISSIONS_FILE
    if not path.exists():
        raise FileNotFoundError("MIMIC-IV admissions file not found at {}".format(path))

    df = pd.read_csv(
        path,
        usecols=list(MIMIC_IV_COLUMNS.keys()),
        parse_dates=["admittime", "dischtime"],
    )

    df = df.rename(columns=MIMIC_IV_COLUMNS)
    return df


def _postprocess_admissions(df):
    """
    Perform minimal, non-clinical postprocessing:
    - enforce dtypes
    - derive length of stay
    - basic sanity checks
    """

    # Enforce integer IDs
    df["subject_id"] = df["subject_id"].astype(int)
    df["hadm_id"] = df["hadm_id"].astype(int)

    # Normalise admission type
    df["admission_type"] = (
        df["admission_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Length of stay in days (can be negative in bad records; handle later)
    df["length_of_stay_days"] = (
        (df["dischtime"] - df["admittime"])
        .dt.total_seconds()
        / 86400.0
    )

    # Basic sanity checks (do not drop rows here)
    if df["subject_id"].isnull().any():
        raise ValueError("Null subject_id detected in admissions data")

    if df["hadm_id"].isnull().any():
        raise ValueError("Null hadm_id detected in admissions data")

    if df.duplicated(subset=["hadm_id"]).any():
        # Duplicates should not occur at this stage
        raise ValueError("Duplicate hadm_id detected in admissions data")

    return df
