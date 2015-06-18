# src/code_mapping/stroke_codelist.py
"""
stroke_codelist.py

Defines stroke-related diagnosis code groupings for secondary-care data.

This module provides:
- Explicit ICD-9 and ICD-10 stroke code families
- Helper functions to determine whether a diagnosis code
  represents a stroke-related signal

This module MUST:
- Be explicit and conservative
- Prefer recall over precision
- Avoid claiming clinical definitiveness

This module MUST NOT:
- Perform phenotype aggregation
- Apply eligibility logic
- Make assumptions about trial protocols
"""

# Removed typing imports for Python 3.4 compatibility


# ---------------------------------------------------------------------
# ICD-9 Stroke Code Families (MIMIC-III proxy)
# ---------------------------------------------------------------------

ICD9_STROKE_PREFIXES = {
    "430",
    "431",
    "432",
    "433",
    "434",
    "435",
    "436",
}


# ---------------------------------------------------------------------
# ICD-10 Stroke Code Families (MIMIC-IV / NHS secondary care analogue)
# ---------------------------------------------------------------------

ICD10_STROKE_PREFIXES = {
    "I60",
    "I61",
    "I62",
    "I63",
    "I64",
    "G45",
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def is_stroke_code(
    diagnosis_code,
    code_system,
):
    """
    Determine whether a diagnosis code represents a stroke-related signal.
    """
    if not diagnosis_code or not code_system:
        return False

    code = diagnosis_code.strip().upper()

    if code_system == "ICD9":
        return _matches_prefix(code, ICD9_STROKE_PREFIXES)

    if code_system == "ICD10":
        return _matches_prefix(code, ICD10_STROKE_PREFIXES)

    raise ValueError("Unsupported code system: {}".format(code_system))


def get_stroke_prefixes(code_system):
    """
    Return the set of stroke-related code prefixes for a given code system.
    """
    if code_system == "ICD9":
        return ICD9_STROKE_PREFIXES

    if code_system == "ICD10":
        return ICD10_STROKE_PREFIXES

    raise ValueError("Unsupported code system: {}".format(code_system))


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _matches_prefix(code, prefixes):
    """
    Check whether a diagnosis code begins with any of the provided prefixes.
    """
    for prefix in prefixes:
        if code.startswith(prefix):
            return True
    return False
