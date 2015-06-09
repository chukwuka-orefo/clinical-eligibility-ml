# src/ingestion/validate_raw.py
"""
validate_raw.py

Validate presence and basic structure of raw MIMIC data files.

Responsibilities:
- Confirm required raw files exist
- Confirm required columns are present
- Surface clear, human-readable errors

This module is designed to:
- Fail early
- Fail clearly
- Support non-technical execution environments (e.g. Flask UI)

This module MUST:
- Perform no data transformation
- Perform no clinical logic
"""

from pathlib import Path

import pandas as pd

from config.paths import (
    MIMIC_III_DIR,
    MIMIC_IV_DIR,
)
from config.settings import ACTIVE_DATASET, Dataset


# ---------------------------------------------------------------------
# Dataset-specific expectations
# ---------------------------------------------------------------------

MIMIC_III_REQUIRED_FILES = {
    "PATIENTS.csv": ["SUBJECT_ID", "GENDER", "DOB"],
    "ADMISSIONS.csv": ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME"],
    "DIAGNOSES_ICD.csv": ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
}

MIMIC_IV_REQUIRED_FILES = {
    "patients.csv": ["subject_id", "gender", "anchor_age", "anchor_year"],
    "admissions.csv": ["subject_id", "hadm_id", "admittime", "dischtime"],
    "diagnoses_icd.csv": ["subject_id", "hadm_id", "icd_code", "icd_version"],
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def validate_raw_data():
    """
    Validate raw data files for the active dataset.

    Raises
    ------
    FileNotFoundError
        If required files are missing.
    ValueError
        If required columns are missing.
    """
    if ACTIVE_DATASET == Dataset.MIMIC_III:
        _validate_dataset(MIMIC_III_DIR, MIMIC_III_REQUIRED_FILES)
    elif ACTIVE_DATASET == Dataset.MIMIC_IV:
        _validate_dataset(MIMIC_IV_DIR, MIMIC_IV_REQUIRED_FILES)
    else:
        raise ValueError("Unsupported dataset: {}".format(ACTIVE_DATASET))


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _validate_dataset(base_dir, required_files):
    if not base_dir.exists():
        raise FileNotFoundError(
            "Raw data directory not found: {}".format(base_dir)
        )

    for filename, required_columns in required_files.items():
        file_path = base_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                "Required raw data file missing: {}".format(file_path)
            )

        _validate_columns(file_path, required_columns)


def _validate_columns(file_path, required_columns):
    try:
        df = pd.read_csv(file_path, nrows=5)
    except Exception as exc:
        raise ValueError(
            "Failed to read raw data file {}: {}".format(file_path, exc)
        )

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            "Raw data file {} is missing required columns: {}".format(
                file_path.name, missing
            )
        )
