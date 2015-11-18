# app/run_engine.py
"""
run_engine.py

Orchestration entry point for the clinical eligibility engine.

Purpose:
- Provide a single function to run the engine end-to-end
- Load study configuration
- Execute eligibility pipeline
- Write outputs for downstream consumption

This module is intentionally UI-agnostic.
It can be called from:
- Flask UI
- batch scripts
- tests or notebooks (internal use)

This module MUST:
- Not contain Flask logic
- Not perform presentation logic
"""

from pathlib import Path
from shutil import copyfile

import pandas as pd

from app.engine.config.study_loader import load_study_config
from app.engine.heuristics.age_rules import apply_age_rules
from app.engine.heuristics.stroke_rules import apply_stroke_rules
from app.engine.heuristics.exclusion_rules import apply_exclusion_rules
from app.engine.models.train import train_models
from app.engine.features.build_feature_matrix import build_feature_matrix
from app.engine.ingestion.run_ingestion import run_ingestion
from app.engine.config.paths import (
    PROCESSED_FEATURES_PATH,
    INTERIM_DATA_DIR,
)
from app.engine.data.reference import REFERENCE_DATA_DIR
from app.engine.utils.logging import get_logger
from app.engine.config.settings import LOG_LEVEL
from app.engine.phenotypes.stroke_phenotype import build_stroke_phenotype
from app.engine.phenotypes.cardiovascular_phenotype import build_cardiovascular_phenotype


# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

LOGGER = get_logger(__name__, LOG_LEVEL)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _interim_data_complete():
    """
    Check whether all required interim data files are present.
    """
    required_files = (
        "patients.csv",
        "admissions.csv",
        "diagnoses.csv",
    )

    for filename in required_files:
        if not (INTERIM_DATA_DIR / filename).exists():
            return False

    return True


def _phenotypes_complete():
    """
    Check whether required phenotype files are present.
    """
    required_files = (
        "stroke_phenotype.csv",
        "cardiovascular_phenotype.csv",
    )

    for filename in required_files:
        if not (INTERIM_DATA_DIR / filename).exists():
            return False

    return True


def _populate_interim_from_reference():
    """
    Populate interim data directory using bundled reference dataset.
    """
    if not INTERIM_DATA_DIR.exists():
        INTERIM_DATA_DIR.mkdir(parents=True)

    for filename in ("patients.csv", "admissions.csv", "diagnoses.csv"):
        src = REFERENCE_DATA_DIR / filename
        dst = INTERIM_DATA_DIR / filename

        if not src.exists():
            raise FileNotFoundError(
                "Reference data missing: {0}".format(src)
            )

        copyfile(src, dst)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_study(study_config_path, output_dir):
    """
    Run the clinical eligibility engine for a given study.
    """

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Load study configuration
    study_config = load_study_config(study_config_path)

    # Ensure interim data exists
    if not _interim_data_complete():
        LOGGER.info(
            "Interim data incomplete, populating from bundled reference dataset"
        )
        _populate_interim_from_reference()

    # Ensure phenotype files exist
    if not _phenotypes_complete():
        LOGGER.info(
            "Phenotype files missing, generating phenotypes"
        )
        build_stroke_phenotype()
        build_cardiovascular_phenotype()

    # Ensure processed features exist
    if not PROCESSED_FEATURES_PATH.exists():
        LOGGER.info(
            "Processed feature matrix not found, building features"
        )
        build_feature_matrix()

    # Train and score
    scored_df = train_models()

    # Apply study-specific heuristics
    scored_df = apply_age_rules(scored_df, study_config)
    scored_df = apply_stroke_rules(scored_df, study_config)
    scored_df = apply_exclusion_rules(scored_df, study_config)

    # Write outputs
    output_path = output_dir / "eligibility_results.csv"
    scored_df.to_csv(output_path, index=False)

    return {
        "study_name": study_config.get("study", {}).get("name", "unknown"),
        "output_file": str(output_path),
        "record_count": len(scored_df),
    }
