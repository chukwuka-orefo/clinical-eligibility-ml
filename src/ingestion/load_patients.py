# src/ingestion/load_patients.py
"""
load_patients.py

Load and normalise patient demographic data from MIMIC-III or MIMIC-IV
and derive age at admission by joining with admissions timestamps.

Responsibilities:
- Read raw PATIENTS tables
- Standardise demographic columns
- Derive age_at_admission using ADMISSIONS
- Handle MIMIC de-identification quirks (age capping)
- Write an interim patients table

This module MUST:
- Operate deterministically
- Contain no phenotype or eligibility logic
- Be safe for both MIMIC-III (ICD-9 era) and MIMIC-IV (ICD-10 era)

This module MUST NOT:
- Infer clinical meaning
- Perform feature engineering beyond demographics
"""

from pathlib import Path

import pandas as pd
import numpy as np

from config.paths import (
    MIMIC_III_DIR,
    MIMIC_IV_DIR,
    ADMISSIONS_INTERIM_PATH,
    INTERIM_DATA_DIR,
    ensure_directories,
)
from config.settings import ACTIVE_DATASET, Dataset


# ---------------------------------------------------------------------
# Dataset-specific filenames
# ---------------------------------------------------------------------

MIMIC_III_PATIENTS_FILE = "PATIENTS.csv"
MIMIC_IV_PATIENTS_FILE = "patients.csv"


# ---------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------

PATIENTS_INTERIM_PATH = INTERIM_DATA_DIR / "patients.csv"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_patients(
    admissions_path=None,
    output_path=None,
):
    """
    Load patient demographics and derive age at admission.

    Parameters
    ----------
    admissions_path : Optional[Path]
        Path to interim admissions table (required for age derivation).
        Defaults to ADMISSIONS_INTERIM_PATH.
    output_path : Optional[Path]
        Path to write interim patients table.
        Defaults to PATIENTS_INTERIM_PATH.

    Returns
    -------
    pd.DataFrame
        Patient-level table with age_at_admission derived per admission.
    """
    ensure_directories()

    if admissions_path is None:
        admissions_path = ADMISSIONS_INTERIM_PATH

    if not admissions_path.exists():
        raise FileNotFoundError(
            "Admissions file not found at {}. Run load_admissions.py first.".format(
                admissions_path
            )
        )

    admissions = pd.read_csv(
        admissions_path,
        parse_dates=["admittime", "dischtime"],
    )

    patients = _load_raw_patients()

    df = _join_and_derive_age(patients, admissions)

    if output_path is None:
        output_path = PATIENTS_INTERIM_PATH

    df.to_csv(output_path, index=False)
    return df


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _load_raw_patients():
    """
    Load raw PATIENTS table for the active dataset and normalise columns.
    """
    if ACTIVE_DATASET == Dataset.MIMIC_III:
        path = MIMIC_III_DIR / MIMIC_III_PATIENTS_FILE
        if not path.exists():
            raise FileNotFoundError(
                "MIMIC-III patients file not found at {}".format(path)
            )

        df = pd.read_csv(
            path,
            usecols=["SUBJECT_ID", "GENDER", "DOB", "DOD"],
            parse_dates=["DOB", "DOD"],
        )

        df = df.rename(
            columns={
                "SUBJECT_ID": "subject_id",
                "GENDER": "sex",
                "DOB": "dob",
                "DOD": "dod",
            }
        )

    elif ACTIVE_DATASET == Dataset.MIMIC_IV:
        path = MIMIC_IV_DIR / MIMIC_IV_PATIENTS_FILE
        if not path.exists():
            raise FileNotFoundError(
                "MIMIC-IV patients file not found at {}".format(path)
            )

        df = pd.read_csv(
            path,
            usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        )

        df = df.rename(
            columns={
                "gender": "sex",
            }
        )

        # MIMIC-IV uses anchor_age / anchor_year instead of DOB
        df["dob"] = pd.to_datetime(
            df["anchor_year"] - df["anchor_age"], format="%Y", errors="coerce"
        )
        df["dod"] = pd.NaT

    else:
        raise ValueError("Unsupported dataset: {}".format(ACTIVE_DATASET))

    return df


def _join_and_derive_age(
    patients,
    admissions,
):
    """
    Join patients to admissions and derive age_at_admission.
    """

    df = admissions.merge(
        patients,
        on="subject_id",
        how="left",
    )

    # Derive age at admission
    df["age_at_admission"] = (
        (df["admittime"] - df["dob"])
        .dt.days
        .div(365.25)
        .astype(float)
    )

    # Handle MIMIC de-identification (age > 89)
    df["age_at_admission"] = df["age_at_admission"].clip(lower=0, upper=90)

    # Normalise sex field
    df["sex"] = (
        df["sex"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"M": "MALE", "F": "FEMALE"})
    )

    # Basic sanity checks
    if df["subject_id"].isnull().any():
        raise ValueError("Null subject_id detected after patient join")

    if df["hadm_id"].isnull().any():
        raise ValueError("Null hadm_id detected after patient join")

    return df[
        [
            "subject_id",
            "hadm_id",
            "sex",
            "age_at_admission",
        ]
    ]
