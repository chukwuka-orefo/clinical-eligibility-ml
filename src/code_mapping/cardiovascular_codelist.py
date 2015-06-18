# src/code_mapping/cardiovascular_codelist.py
"""
cardiovascular_codelist.py

Defines cardiovascular-related diagnosis code groupings for secondary-care data.

This module provides:
- Explicit ICD-9 and ICD-10 cardiovascular code families
- Helper functions to determine whether a diagnosis code
  represents cardiovascular disease context

This module MUST:
- Be conservative and transparent
- Treat codes as signals, not definitive diagnoses
- Support eligibility context, not drive eligibility alone

This module MUST NOT:
- Apply phenotype aggregation
- Apply eligibility logic
- Make assumptions about specific trial protocols
"""

# Removed typing imports for Python 3.4 compatibility


# ---------------------------------------------------------------------
# ICD-9 Cardiovascular Code Families (MIMIC-III proxy)
# ---------------------------------------------------------------------

ICD9_CARDIOVASCULAR_PREFIXES = {
    "390",
    "391",
    "392",
    "393",
    "394",
    "395",
    "396",
    "397",
    "398",
    "401",
    "402",
    "403",
    "404",
    "405",
    "410",
    "411",
    "412",
    "413",
    "414",
    "415",
    "416",
    "417",
    "420",
    "421",
    "422",
    "423",
    "424",
    "425",
    "426",
    "427",
    "428",
    "429",
}


# ---------------------------------------------------------------------
# ICD-10 Cardiovascular Code Families (NHS analogue / MIMIC-IV)
# ---------------------------------------------------------------------

ICD10_CARDIOVASCULAR_PREFIXES = {
    "I00",
    "I01",
    "I02",
    "I03",
    "I04",
    "I05",
    "I06",
    "I07",
    "I08",
    "I09",
    "I10",
    "I11",
    "I12",
    "I13",
    "I15",
    "I20",
    "I21",
    "I22",
    "I23",
    "I24",
    "I25",
    "I26",
    "I27",
    "I28",
    "I30",
    "I31",
    "I32",
    "I33",
    "I34",
    "I35",
    "I36",
    "I37",
    "I38",
    "I39",
    "I40",
    "I41",
    "I42",
    "I43",
    "I44",
    "I45",
    "I46",
    "I47",
    "I48",
    "I49",
    "I50",
    "I51",
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def is_cardiovascular_code(diagnosis_code, code_system):
    """
    Determine whether a diagnosis code represents a cardiovascular-related signal.

    Parameters
    ----------
    diagnosis_code : str
        Raw diagnosis code (ICD-9 or ICD-10).
    code_system : str
        Code system identifier ("ICD9" or "ICD10").

    Returns
    -------
    bool
        True if the code is cardiovascular-related, False otherwise.
    """
    if not diagnosis_code or not code_system:
        return False

    code = diagnosis_code.strip().upper()

    if code_system == "ICD9":
        return _matches_prefix(code, ICD9_CARDIOVASCULAR_PREFIXES)

    if code_system == "ICD10":
        return _matches_prefix(code, ICD10_CARDIOVASCULAR_PREFIXES)

    raise ValueError("Unsupported code system: {}".format(code_system))


def get_cardiovascular_prefixes(code_system):
    """
    Return the set of cardiovascular-related code prefixes for a given code system.

    Parameters
    ----------
    code_system : str
        "ICD9" or "ICD10"

    Returns
    -------
    Set[str]
        Set of code prefixes used to identify cardiovascular signals.
    """
    if code_system == "ICD9":
        return ICD9_CARDIOVASCULAR_PREFIXES

    if code_system == "ICD10":
        return ICD10_CARDIOVASCULAR_PREFIXES

    raise ValueError("Unsupported code system: {}".format(code_system))


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _matches_prefix(code, prefixes):
    """
    Check whether a diagnosis code begins with any of the provided prefixes.

    Uses prefix matching to reflect real-world clinical coding variability.

    Parameters
    ----------
    code : str
        Diagnosis code
    prefixes : Iterable[str]
        Code prefixes to match against

    Returns
    -------
    bool
    """
    for prefix in prefixes:
        if code.startswith(prefix):
            return True
    return False
