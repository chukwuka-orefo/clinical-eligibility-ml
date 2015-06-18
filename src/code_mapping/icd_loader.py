# src/code_mapping/icd_loader.py
"""
icd_loader.py

Utility helpers for handling ICD diagnosis codes across datasets.

Purpose:
- Provide a single, explicit place where ICD-9 vs ICD-10
  differences are handled.
- Normalize diagnosis codes for downstream codelists.
- Avoid dataset-specific logic leaking into phenotype or heuristic layers.

This module MUST:
- Be deterministic
- Contain no clinical meaning
- Contain no eligibility logic

This module MUST NOT:
- Decide whether a code represents stroke, CVD, etc.
- Perform aggregation
- Access data files
"""

# Removed typing imports (not allowed in Python 3.4)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def normalise_icd_code(code):
    """
    Normalise a raw ICD diagnosis code.

    Parameters
    ----------
    code : Optional[str]
        Raw ICD code (ICD-9 or ICD-10).

    Returns
    -------
    Optional[str]
        Normalised ICD code (uppercase, stripped), or None.
    """
    if code is None:
        return None

    if not isinstance(code, str):
        return None

    code = code.strip().upper()

    if code == "":
        return None

    return code


def infer_code_system(code_system):
    """
    Validate and normalise ICD code system identifier.

    Parameters
    ----------
    code_system : Optional[str]
        Expected values: "ICD9" or "ICD10".

    Returns
    -------
    Optional[str]
        Normalised code system string, or None.

    Raises
    ------
    ValueError
        If code_system is not recognised.
    """
    if code_system is None:
        return None

    code_system = code_system.strip().upper()

    if code_system in {"ICD9", "ICD10"}:
        return code_system

    raise ValueError("Unsupported ICD code system: {}".format(code_system))


def split_icd_prefix(code, length=3):
    """
    Extract the prefix used for ICD grouping.

    Parameters
    ----------
    code : Optional[str]
        Normalised ICD code.
    length : int
        Number of leading characters to use as prefix.

    Returns
    -------
    Optional[str]
        ICD prefix, or None.
    """
    if code is None:
        return None

    if len(code) < length:
        return None

    return code[:length]
