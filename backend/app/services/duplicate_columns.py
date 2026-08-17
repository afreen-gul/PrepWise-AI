"""Exact and near-duplicate column detection (value-based, not name-based)."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from app.schemas.cleaning import CleaningLogEntry

NEAR_DUPLICATE_MIN_SIMILARITY = 0.95
_MAX_NEAR_DUP_CANDIDATES_PER_BUCKET = 40


def column_value_fingerprint(series: pd.Series) -> str:
    """Stable row-wise signature; NaN positions must match for identical columns."""
    normalized = series.astype(object).where(series.notna(), other="__NA__").astype(str)
    return normalized.str.cat(sep="|")


def columns_are_exact_duplicates(a: pd.Series, b: pd.Series) -> bool:
    """True when values match in every row, including aligned missing positions."""
    if len(a) != len(b):
        return False
    aligned = a.reset_index(drop=True)
    other = b.reset_index(drop=True)
    both_na = aligned.isna() & other.isna()
    equal = aligned.eq(other) | both_na
    return bool(equal.all())


def row_agreement_ratio(a: pd.Series, b: pd.Series) -> float:
    """Fraction of rows where values match (including both missing)."""
    if len(a) == 0:
        return 1.0
    aligned = a.reset_index(drop=True)
    other = b.reset_index(drop=True)
    both_na = aligned.isna() & other.isna()
    matches = aligned.eq(other) | both_na
    return float(matches.sum()) / len(aligned)


def find_exact_duplicate_column_groups(df: pd.DataFrame) -> dict[str, str]:
    """Map duplicate column name -> original column (first by dataframe column order)."""
    col_order = [str(c) for c in df.columns]
    order_index = {name: idx for idx, name in enumerate(col_order)}

    buckets: dict[str, list[str]] = {}
    for column in df.columns:
        col_name = str(column)
        key = column_value_fingerprint(df[column])
        buckets.setdefault(key, []).append(col_name)

    duplicate_of: dict[str, str] = {}
    for names in buckets.values():
        if len(names) < 2:
            continue
        canonical = min(names, key=lambda n: order_index[n])
        for name in names:
            if name != canonical:
                duplicate_of[name] = canonical
    return duplicate_of


def list_exact_duplicate_pairs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """One record per redundant column."""
    duplicate_of = find_exact_duplicate_column_groups(df)
    return [
        {
            "duplicate_column": dup,
            "original_column": orig,
            "similarity": 100.0,
            "action": "detected",
        }
        for dup, orig in duplicate_of.items()
    ]


def detect_potentially_redundant_columns(
    df: pd.DataFrame,
    *,
    exclude_columns: set[str] | None = None,
) -> list[dict[str, Any]]:
    """High similarity but not exact — report only, do not remove."""
    exclude = exclude_columns or set()
    col_names = [str(c) for c in df.columns if str(c) not in exclude]
    if len(col_names) < 2:
        return []

    # Pre-group by dtype kind and missing count to limit pairwise work.
    buckets: dict[tuple[str, int], list[str]] = {}
    for name in col_names:
        series = df[name]
        dtype_key = str(series.dtype)
        null_count = int(series.isna().sum())
        buckets.setdefault((dtype_key, null_count), []).append(name)

    findings: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for names in buckets.values():
        if len(names) < 2:
            continue
        if len(names) > _MAX_NEAR_DUP_CANDIDATES_PER_BUCKET:
            continue
        for col_a, col_b in combinations(names, 2):
            orig, dup = (col_a, col_b) if col_a < col_b else (col_b, col_a)
            pair_key = (orig, dup)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            a = df[col_a]
            b = df[col_b]
            if columns_are_exact_duplicates(a, b):
                continue
            ratio = row_agreement_ratio(a, b)
            if NEAR_DUPLICATE_MIN_SIMILARITY <= ratio < 1.0:
                findings.append(
                    {
                        "column_a": col_a,
                        "column_b": col_b,
                        "similarity": round(ratio * 100.0, 2),
                        "status": "potentially_redundant",
                    }
                )
    return findings


def _log_duplicate_entry(
    log: list[CleaningLogEntry],
    operation: str,
    *,
    column: str | None,
    message: str,
    **details: Any,
) -> None:
    log.append(
        CleaningLogEntry(
            operation=operation,
            column=column,
            details=details,
            message=message,
        )
    )


def process_duplicate_columns(
    df: pd.DataFrame,
    log: list[CleaningLogEntry],
    *,
    remove: bool,
) -> pd.DataFrame:
    """Detect exact duplicates; optionally drop redundant columns (keep first in order)."""
    duplicate_of = find_exact_duplicate_column_groups(df)
    near = detect_potentially_redundant_columns(
        df, exclude_columns=set(duplicate_of.keys())
    )

    pairs = list_exact_duplicate_pairs(df)
    if pairs or near:
        _log_duplicate_entry(
            log,
            "duplicate_column_detection",
            column=None,
            detected_count=len(pairs),
            potentially_redundant_count=len(near),
            duplicate_columns=pairs,
            potentially_redundant=near,
            message=(
                f"Detected {len(pairs)} exact duplicate column(s)"
                + (f"; {len(near)} potentially redundant pair(s)" if near else "")
                + "."
            ),
        )

    for dup, orig in duplicate_of.items():
        reason = (
            f"Column contains identical values to {orig} and provides no "
            "additional information."
        )
        if remove:
            _log_duplicate_entry(
                log,
                "duplicate_column_removal",
                column=dup,
                issue="Duplicate Column",
                duplicate_of=orig,
                original_column=orig,
                duplicate_column=dup,
                similarity=100.0,
                action="Removed",
                reason=reason,
                message=f"Removed duplicate column '{dup}' (identical to '{orig}').",
            )
        else:
            _log_duplicate_entry(
                log,
                "duplicate_column_retained",
                column=dup,
                issue="Duplicate Column",
                duplicate_of=orig,
                original_column=orig,
                duplicate_column=dup,
                similarity=100.0,
                action="Retained (option off)",
                reason=reason,
                message=f"Duplicate column '{dup}' identical to '{orig}' (not removed).",
            )

    if not remove or not duplicate_of:
        return df

    to_drop = [c for c in df.columns if str(c) in duplicate_of]
    return df.drop(columns=to_drop)
