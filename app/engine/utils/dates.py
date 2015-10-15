# src/utils/dates.py
"""
dates.py

Date and time utility functions for Trial Eligibility ML.

Design goals:
- No external dependencies
- Deterministic behaviour
- Safe handling of missing or malformed dates
- Reusable across ingestion, phenotypes, and heuristics

This module MUST:
- Contain only pure utility functions
- Perform no I/O
- Contain no clinical or eligibility logic
"""

import pandas as pd


# ---------------------------------------------------------------------
# Age derivation
# ---------------------------------------------------------------------

def compute_age_at_event(
    date_of_birth,
    event_time,
):
    """
    Compute age in years at the time of an event.

    Parameters
    ----------
    date_of_birth : pd.Timestamp
        Date of birth.
    event_time : pd.Timestamp
        Event timestamp (e.g. admission time).

    Returns
    -------
    Optional[float]
        Age in years, or None if inputs are invalid.
    """
    if pd.isna(date_of_birth) or pd.isna(event_time):
        return None

    if event_time < date_of_birth:
        # Defensive: malformed or de-identified data
        return None

    delta_days = (event_time - date_of_birth).days
    return delta_days / 365.25


def clip_age(
    age,
    min_age=0,
    max_age=120,
):
    """
    Clip age to plausible bounds.

    Parameters
    ----------
    age : Optional[float]
        Age value to clip.
    min_age : int
        Minimum plausible age.
    max_age : int
        Maximum plausible age.

    Returns
    -------
    Optional[float]
        Clipped age, or None if input is None.
    """
    if age is None:
        return None

    return max(min(age, max_age), min_age)


# ---------------------------------------------------------------------
# Length-of-stay and duration helpers
# ---------------------------------------------------------------------

def compute_duration_days(
    start_time,
    end_time,
):
    """
    Compute duration in days between two timestamps.

    Parameters
    ----------
    start_time : pd.Timestamp
        Start time.
    end_time : pd.Timestamp
        End time.

    Returns
    -------
    Optional[float]
        Duration in days, or None if inputs are invalid.
    """
    if pd.isna(start_time) or pd.isna(end_time):
        return None

    if end_time < start_time:
        return None

    return (end_time - start_time).total_seconds() / 86400.0


# ---------------------------------------------------------------------
# Defensive checks
# ---------------------------------------------------------------------

def is_valid_timestamp(ts):
    """
    Check whether a value is a valid pandas timestamp.

    Parameters
    ----------
    ts : pd.Timestamp

    Returns
    -------
    bool
        True if valid timestamp, False otherwise.
    """
    return isinstance(ts, pd.Timestamp) and not pd.isna(ts)
