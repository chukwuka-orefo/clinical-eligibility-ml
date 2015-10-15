# config/study_loader.py
"""
study_loader.py

Load and validate study-specific eligibility configuration from YAML.

Purpose:
- Provide a single, safe entry point for study configuration
- Apply defaults where fields are missing
- Keep configuration separate from clinical and ML logic

This module MUST:
- Load YAML from disk
- Validate structure defensively
- Return a plain Python dictionary

This module MUST NOT:
- Apply eligibility logic
- Contain study-specific assumptions
- Mutate global state
"""

from pathlib import Path

import yaml

from utils.checks import require_columns


# ---------------------------------------------------------------------
# Defaults (mirrors Phase A thresholds)
# ---------------------------------------------------------------------

DEFAULT_STUDY_CONFIG = {
    "age": {
        "min": 18,
        "max": 85,
        "hard_exclude": 90,
    },
    "stroke_signal": {
        "min_code_count": 1,
        "require_any_signal": True,
        "prefer_primary_dx": True,
    },
    "cardiovascular_context": {
        "min_code_count": 1,
        "required": False,
    },
    "admission": {
        "emergency_only": False,
    },
    "ml_scoring": {
        "enabled": False,
        "min_score": 0.0,
    },
    "exclusions": {
        "exclude_without_stroke_signal": True,
        "exclude_if_age_above_hard_limit": True,
    },
    "screening": {
        "default_k_values": [25, 50, 100, 200],
    },
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_study_config(path):
    """
    Load study configuration from YAML and apply defaults.

    Parameters
    ----------
    path : Path
        Path to study_config.yaml.

    Returns
    -------
    Dict[str, Any]
        Validated study configuration dictionary.
    """
    if not path.exists():
        raise FileNotFoundError("Study config file not found: {}".format(path))

    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError("Study config file is empty")

    # Basic structure check
    require_columns(
        raw_config,
        ["study"],
        context="Study config",
    )

    config = _apply_defaults(raw_config, DEFAULT_STUDY_CONFIG)

    return config


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _apply_defaults(
    user_config,
    defaults,
):
    """
    Recursively apply defaults to user config.

    User-provided values always override defaults.
    """
    merged = {}

    for key, default_value in defaults.items():
        if key not in user_config:
            merged[key] = default_value
        else:
            if isinstance(default_value, dict) and isinstance(user_config[key], dict):
                merged[key] = _apply_defaults(user_config[key], default_value)
            else:
                merged[key] = user_config[key]

    # Preserve top-level metadata (e.g. study name)
    for key, value in user_config.items():
        if key not in merged:
            merged[key] = value

    return merged
