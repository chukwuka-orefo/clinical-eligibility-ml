# src/config/paths.py
"""
paths.py

Centralised definition of filesystem paths for the Trial Eligibility ML project.

This module should contain NO clinical logic and NO dataset-specific assumptions.
All other modules import paths from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

# Assumes this file lives in src/config/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------
# Raw dataset locations (dataset-specific subfolders live here)
# ---------------------------------------------------------------------

MIMIC_III_DIR = RAW_DATA_DIR / "mimic-iii"
MIMIC_IV_DIR = RAW_DATA_DIR / "mimic-iv"

# ---------------------------------------------------------------------
# Interim outputs
# ---------------------------------------------------------------------

ADMISSIONS_INTERIM_PATH = INTERIM_DATA_DIR / "admissions.csv"
DIAGNOSES_INTERIM_PATH = INTERIM_DATA_DIR / "diagnoses.csv"
ADMISSION_DIAGNOSES_INTERIM_PATH = INTERIM_DATA_DIR / "admission_diagnoses.csv"

# ---------------------------------------------------------------------
# Processed modelling dataset
# ---------------------------------------------------------------------

PROCESSED_FEATURES_PATH = PROCESSED_DATA_DIR / "trial_eligibility_features.csv"

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def ensure_directories():
    """
    Create required data directories if they do not exist.
    Safe to call multiple times.
    """
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
