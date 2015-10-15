# src/config/settings.py
"""
settings.py

Global configuration for Trial Eligibility ML.

This module controls:
- dataset selection
- code system assumptions
- run-time flags

It should NOT contain paths (see paths.py) or modelling logic.
"""

from enum import Enum

# ---------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------

class Dataset(Enum):
    MIMIC_III = "mimic_iii"
    MIMIC_IV = "mimic_iv"

# Default dataset (can be changed later)
ACTIVE_DATASET = Dataset.MIMIC_IV

# ---------------------------------------------------------------------
# Clinical code system configuration
# ---------------------------------------------------------------------

class CodeSystem(Enum):
    ICD9 = "icd9"
    ICD10 = "icd10"

# Map dataset → code system
DATASET_CODE_SYSTEM = {
    Dataset.MIMIC_III: CodeSystem.ICD9,
    Dataset.MIMIC_IV: CodeSystem.ICD10,
}

ACTIVE_CODE_SYSTEM = DATASET_CODE_SYSTEM[ACTIVE_DATASET]

# ---------------------------------------------------------------------
# Modelling grain
# ---------------------------------------------------------------------

# Unit of analysis: hospital admission (episode-level)
MODELLING_GRAIN = "admission"

# ---------------------------------------------------------------------
# Run-time flags
# ---------------------------------------------------------------------

# Whether to allow optional features that may risk leakage
ALLOW_TEMPORAL_FEATURES = False

# Whether to include elective admissions in modelling
INCLUDE_ELECTIVE_ADMISSIONS = True

# ---------------------------------------------------------------------
# Logging / verbosity
# ---------------------------------------------------------------------

LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

def validate_settings():
    """
    Basic sanity checks to ensure configuration consistency.
    """
    if ACTIVE_DATASET not in DATASET_CODE_SYSTEM:
        raise ValueError("No code system defined for dataset {}".format(ACTIVE_DATASET))

    if ACTIVE_CODE_SYSTEM != DATASET_CODE_SYSTEM[ACTIVE_DATASET]:
        raise ValueError(
            "ACTIVE_CODE_SYSTEM does not match ACTIVE_DATASET mapping"
        )
