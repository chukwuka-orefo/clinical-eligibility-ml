# run_engine.py
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

import pandas as pd

from app.engine.config.study_loader import load_study_config
from app.engine.heuristics.age_rules import apply_age_rules
from app.engine.heuristics.stroke_rules import apply_stroke_rules
from app.engine.heuristics.exclusion_rules import apply_exclusion_rules
from app.engine.models.train import train_models
from app.engine.features.build_feature_matrix import build_feature_matrix
from app.engine.config.paths import PROCESSED_FEATURES_PATH
# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_study(
    study_config_path,
    output_dir,
):
    """
    Run the clinical eligibility engine for a given study.

    Parameters
    ----------
    study_config_path : Path
        Path to study_config.yaml.
    output_dir : Path
        Directory where outputs will be written.

    Returns
    -------
    Dict[str, Any]
        Summary information about the run.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load study configuration
    # -----------------------------------------------------------------
    study_config = load_study_config(study_config_path)

    # -----------------------------------------------------------------
    # Train and score using existing pipeline
    # -----------------------------------------------------------------
    if not PROCESSED_FEATURES_PATH.exists():
        build_feature_matrix()
        
    scored_df = train_models()

    # -----------------------------------------------------------------
    # Apply study-specific heuristics
    # -----------------------------------------------------------------
    scored_df = apply_age_rules(scored_df, study_config)
    scored_df = apply_stroke_rules(scored_df, study_config)
    scored_df = apply_exclusion_rules(scored_df, study_config)

    # -----------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------
    output_path = output_dir / "eligibility_results.csv"
    scored_df.to_csv(output_path, index=False)

    return {
        "study_name": study_config.get("study", {}).get("name", "unknown"),
        "output_file": str(output_path),
        "record_count": len(scored_df),
    }
