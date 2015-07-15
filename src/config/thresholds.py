# src/config/thresholds.py
"""
thresholds.py

Centralised definition of heuristic thresholds for Trial Eligibility ML.

This module contains ONLY constants.
No logic, no data access, no imports from other project modules.

All heuristic rules must reference thresholds defined here so that:
- assumptions are explicit
- changes are auditable
- experiments are reproducible
"""

# ---------------------------------------------------------------------
# Age thresholds (years)
# ---------------------------------------------------------------------

# Minimum age for inclusion in adult stroke trials
MIN_ELIGIBLE_AGE: int = 18

# Upper age bound for standard inclusion
# Chosen to be permissive and recall-oriented
MAX_ELIGIBLE_AGE: int = 85

# Hard exclusion age
# Admissions above this age are excluded entirely
MAX_EXCLUSION_AGE: int = 90


# ---------------------------------------------------------------------
# Stroke signal thresholds
# ---------------------------------------------------------------------

# Minimum number of stroke-related diagnosis codes
# required to consider an admission as having a stroke signal
MIN_STROKE_CODE_COUNT: int = 1

# Whether primary diagnosis stroke signal should be preferred
# (used for analysis, not enforced as a hard rule)
PREFER_PRIMARY_STROKE_DIAGNOSIS: bool = True


# ---------------------------------------------------------------------
# Cardiovascular context thresholds
# ---------------------------------------------------------------------

# Minimum number of cardiovascular diagnosis codes
# to flag supporting cardiovascular context
MIN_CVD_CODE_COUNT: int = 1


# ---------------------------------------------------------------------
# Admission context thresholds
# ---------------------------------------------------------------------

# Whether emergency admissions are preferred over elective
# (elective inclusion controlled in settings.py)
PREFER_EMERGENCY_ADMISSIONS: bool = True


# ---------------------------------------------------------------------
# Screening workload defaults (for evaluation)
# ---------------------------------------------------------------------

# Default K values for ranking evaluation
# Represent typical manual screening capacities
DEFAULT_SCREENING_K_VALUES = [25, 50, 100, 200]


# ---------------------------------------------------------------------
# Sanity bounds (defensive)
# ---------------------------------------------------------------------

# Absolute lower and upper bounds for derived age values
MIN_POSSIBLE_AGE: int = 0
MAX_POSSIBLE_AGE: int = 120
