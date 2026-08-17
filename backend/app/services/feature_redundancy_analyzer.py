"""Phase 6.2–6.4 — exact duplicates, correlation, categorical redundancy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.schemas.feature_selection import CorrelationPairRow, FeatureQualityRow
from app.services.duplicate_columns import find_exact_duplicate_column_groups
from app.services.feature_selection_config import CORRELATION_THRESHOLD


def _prefer_between_pair(
    a: str,
    b: str,
    *,
    quality_by_name: dict[str, FeatureQualityRow],
    mi_by_name: dict[str, float] | None,
    generated_meta: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    """Return preferred feature (or None) and reason."""
    qa = quality_by_name.get(a)
    qb = quality_by_name.get(b)
    if qa and qa.is_identifier and not (qb and qb.is_identifier):
        return b, f"{a} looks like an identifier; prefer {b}."
    if qb and qb.is_identifier and not (qa and qa.is_identifier):
        return a, f"{b} looks like an identifier; prefer {a}."

    if qa and qa.is_exact_duplicate:
        return qa.duplicate_of or b, f"{a} is an exact duplicate."
    if qb and qb.is_exact_duplicate:
        return qb.duplicate_of or a, f"{b} is an exact duplicate."

    if qa and qb:
        miss_diff = abs(qa.missing_pct - qb.missing_pct)
        if miss_diff >= 10.0:
            preferred = a if qa.missing_pct < qb.missing_pct else b
            other = b if preferred == a else a
            return preferred, (
                f"{other} has significantly higher missingness; prefer {preferred}."
            )

    a_gen = a in generated_meta
    b_gen = b in generated_meta
    if a_gen != b_gen:
        # Mark relationship; do not auto-delete — caller uses REVIEW.
        original = b if a_gen else a
        generated = a if a_gen else b
        return None, (
            f"{generated} appears generated from / related to {original}; "
            "review redundancy rather than auto-removing."
        )

    if mi_by_name:
        mi_a = mi_by_name.get(a)
        mi_b = mi_by_name.get(b)
        if mi_a is not None and mi_b is not None and abs(mi_a - mi_b) >= 0.02:
            preferred = a if mi_a > mi_b else b
            return preferred, (
                f"{preferred} has a stronger mutual-information relationship "
                "with the target."
            )

    return None, (
        "Features contain overlapping information; keep both under REVIEW "
        "until a clearer preference is available."
    )


def analyze_correlation_pairs(
    df: pd.DataFrame,
    *,
    quality_rows: list[FeatureQualityRow],
    target_column: str | None = None,
    mi_by_name: dict[str, float] | None = None,
    generated_meta: dict[str, dict[str, Any]] | None = None,
    threshold: float = CORRELATION_THRESHOLD,
) -> list[CorrelationPairRow]:
    """Find highly correlated numerical pairs (Pearson)."""
    generated_meta = generated_meta or {}
    quality_by_name = {q.feature: q for q in quality_rows}

    numeric_cols = [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and (target_column is None or str(c) != target_column)
        and not (quality_by_name.get(str(c)) and quality_by_name[str(c)].is_constant)
    ]
    if len(numeric_cols) < 2:
        return []

    corr = df[numeric_cols].corr(method="pearson")
    pairs: list[CorrelationPairRow] = []
    seen: set[tuple[str, str]] = set()

    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1 :]:
            val = corr.loc[a, b]
            if pd.isna(val):
                continue
            abs_corr = float(abs(val))
            if abs_corr < threshold:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            preferred, reason = _prefer_between_pair(
                a,
                b,
                quality_by_name=quality_by_name,
                mi_by_name=mi_by_name,
                generated_meta=generated_meta,
            )
            pairs.append(
                CorrelationPairRow(
                    feature_a=a,
                    feature_b=b,
                    correlation=round(float(val), 4),
                    recommendation=(
                        f"Prefer {preferred}; review redundancy"
                        if preferred
                        else "Review redundancy"
                    ),
                    preferred_feature=preferred,
                    reason=reason,
                )
            )

    pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
    return pairs


def categorical_exact_redundancy_notes(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
) -> list[str]:
    """Cheap categorical exact-duplicate notes (no expensive near-dup search)."""
    duplicate_of = find_exact_duplicate_column_groups(df)
    notes: list[str] = []
    for dup, orig in duplicate_of.items():
        if target_column and dup == target_column:
            continue
        if not pd.api.types.is_numeric_dtype(df[dup]):
            notes.append(
                f"`{dup}` is an exact duplicate of `{orig}` "
                "(categorical/object values match row-for-row)."
            )
    return notes
