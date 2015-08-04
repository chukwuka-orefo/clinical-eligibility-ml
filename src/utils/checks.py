# src/utils/checks.py
"""
checks.py

Centralised sanity and validation helpers for Trial Eligibility ML.

Design goals:
- Fail early with clear messages
- Avoid deep stack traces for user-facing workflows
- Keep checks reusable and explicit
- No I/O, no clinical logic, no modelling logic

This module SHOULD:
- Be used by ingestion, features, and evaluation layers
- Raise ValueError with human-readable messages

This module MUST NOT:
- Mutate data
- Perform transformations
- Import project-specific business logic
"""

import pandas as pd


# ---------------------------------------------------------------------
# DataFrame structure checks
# ---------------------------------------------------------------------

def require_columns(
    df,
    required_columns,
    context="DataFrame",
):
    """
    Ensure a DataFrame contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : Iterable[str]
        Column names that must be present.
    context : str
        Short description of where the check is applied.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    required = set(required_columns)
    present = set(df.columns)

    missing = required - present
    if missing:
        raise ValueError(
            "{} is missing required columns: {}".format(context, sorted(missing))
        )


def forbid_columns(
    df,
    forbidden_columns,
    context="DataFrame",
):
    """
    Ensure a DataFrame does NOT contain forbidden columns.

    Useful for guarding against leakage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    forbidden_columns : Iterable[str]
        Column names that must not be present.
    context : str
        Short description of where the check is applied.

    Raises
    ------
    ValueError
        If any forbidden columns are present.
    """
    forbidden = set(forbidden_columns)
    present = set(df.columns)

    found = forbidden & present
    if found:
        raise ValueError(
            "{} contains forbidden columns: {}".format(context, sorted(found))
        )


# ---------------------------------------------------------------------
# Identifier checks
# ---------------------------------------------------------------------

def require_unique(
    df,
    column,
    context="DataFrame",
):
    """
    Ensure a column contains unique values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    column : str
        Column that must be unique.
    context : str
        Short description of where the check is applied.

    Raises
    ------
    ValueError
        If duplicates are found.
    """
    if column not in df.columns:
        raise ValueError("{} missing column: {}".format(context, column))

    if df[column].duplicated().any():
        raise ValueError(
            "{} contains duplicate values in column '{}'".format(context, column)
        )


def require_non_null(
    df,
    column,
    context="DataFrame",
):
    """
    Ensure a column contains no null values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    column : str
        Column that must not contain nulls.
    context : str
        Short description of where the check is applied.

    Raises
    ------
    ValueError
        If null values are present.
    """
    if column not in df.columns:
        raise ValueError("{} missing column: {}".format(context, column))

    if df[column].isnull().any():
        raise ValueError(
            "{} contains null values in column '{}'".format(context, column)
        )


# ---------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------

def require_boolean(
    df,
    column,
    context="DataFrame",
):
    """
    Ensure a column is boolean dtype.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    column : str
        Column expected to be boolean.
    context : str
        Short description of where the check is applied.

    Raises
    ------
    TypeError
        If column is not boolean.
    """
    if column not in df.columns:
        raise ValueError("{} missing column: {}".format(context, column))

    if not pd.api.types.is_bool_dtype(df[column]):
        raise TypeError(
            "{} column '{}' must be boolean".format(context, column)
        )


def require_numeric(
    df,
    column,
    context="DataFrame",
):
    """
    Ensure a column is numeric dtype.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    column : str
        Column expected to be numeric.
    context : str
        Short description of where the check is applied.

    Raises
    ------
    TypeError
        If column is not numeric.
    """
    if column not in df.columns:
        raise ValueError("{} missing column: {}".format(context, column))

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(
            "{} column '{}' must be numeric".format(context, column)
        )


# ---------------------------------------------------------------------
# Range and value checks
# ---------------------------------------------------------------------

def require_non_negative(
    df,
    column,
    context="DataFrame",
):
    """
    Ensure a numeric column contains no negative values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    column : str
        Numeric column expected to be non-negative.
    context : str
        Short description of where the check is applied.

    Raises
    ------
    ValueError
        If negative values are present.
    """
    require_numeric(df, column, context=context)

    if (df[column] < 0).any():
        raise ValueError(
            "{} column '{}' contains negative values".format(context, column)
        )


def require_probability(
    df,
    column,
    context="DataFrame",
):
    """
    Ensure a column represents probabilities in [0, 1].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    column : str
        Column expected to contain probabilities.
    context : str
        Short description of where the check is applied.

    Raises
    ------
    ValueError
        If values fall outside [0, 1].
    """
    require_numeric(df, column, context=context)

    if ((df[column] < 0) | (df[column] > 1)).any():
        raise ValueError(
            "{} column '{}' contains values outside [0, 1]".format(context, column)
        )
