"""
run_ingestion.py

Orchestration entry point for ingestion pipeline.

Purpose:
- Generate interim data tables required by feature engineering
- Provide a single callable ingestion entry point for product use

This module MUST:
- Call existing ingestion functions
- Write interim CSVs to disk
- Contain no feature engineering or modelling logic
"""

from app.engine.ingestion.load_patients import load_patients
from app.engine.ingestion.load_admissions import load_admissions
from app.engine.ingestion.load_diagnoses import load_diagnoses
from app.engine.config.paths import INTERIM_DATA_DIR


def run_ingestion() -> None:
    """
    Run ingestion pipeline to produce interim data files.
    """
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    load_patients()
    load_admissions()
    load_diagnoses()
