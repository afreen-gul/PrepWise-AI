"""Phase 5.5 — Validate and finalize generated features.

Never deletes original user columns. Removes only bad *generated* features.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.feature_engineering import (
    DatetimeIssueFlag,
    GeneratedFeatureMeta,
    RemovedFeatureMeta,
)
from app.services.duplicate_columns import column_value_fingerprint
from app.services.feature_engineering_config import (
    GENERATED_FEATURE_MAX_MISSING_PERCENT,
)

_MONTH_RE = re.compile(r"_Month$", re.IGNORECASE)
_DOW_RE = re.compile(r"_DayOfWeek$", re.IGNORECASE)
_WEEKEND_RE = re.compile(r"_IsWeekend$", re.IGNORECASE)
_DAY_RE = re.compile(r"_Day$", re.IGNORECASE)


class FeatureValidationError(Exception):
    """Raised when validation finds a fatal integrity issue."""


def _series_stats_missing_pct(series: pd.Series) -> float:
    n = len(series)
    if n == 0:
        return 0.0
    return float(series.isna().sum()) / n * 100.0


def validate_and_finalize_features(
    df: pd.DataFrame,
    *,
    original_columns: list[str],
    generated_meta: list[GeneratedFeatureMeta],
    expected_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop bad generated columns; protect originals; enforce row count."""
    working = df.copy()
    if len(working) != expected_rows:
        raise FeatureValidationError(
            f"Row count mismatch before validation: expected {expected_rows}, "
            f"got {len(working)}."
        )

    original_set = set(original_columns)
    for col in original_columns:
        if col not in working.columns:
            raise FeatureValidationError(
                f"Original column '{col}' was removed during feature engineering."
            )

    generated_names = [m.feature for m in generated_meta if m.feature in working.columns]
    meta_by_name = {m.feature: m for m in generated_meta}
    removed: list[RemovedFeatureMeta] = []
    issues: list[DatetimeIssueFlag] = []
    kept_meta: list[GeneratedFeatureMeta] = []

    # --- Infinite values → NaN ---
    for name in list(generated_names):
        if name not in working.columns:
            continue
        series = working[name]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        arr = series.to_numpy(dtype=float, na_value=np.nan)
        inf_mask = np.isinf(arr)
        inf_count = int(np.nansum(inf_mask))
        if inf_count:
            working[name] = series.replace([np.inf, -np.inf], np.nan)
            issues.append(
                DatetimeIssueFlag(
                    issue="infinite_values",
                    columns=[name],
                    count=inf_count,
                    message=(
                        f"{inf_count} infinite value(s) in '{name}' replaced with NaN."
                    ),
                )
            )

    # --- Constant generated features ---
    for name in list(generated_names):
        if name not in working.columns or name in original_set:
            continue
        if int(working[name].nunique(dropna=True)) <= 1:
            working = working.drop(columns=[name])
            meta = meta_by_name.get(name)
            removed.append(
                RemovedFeatureMeta(
                    feature=name,
                    source=meta.source if meta else None,
                    category=meta.category if meta else None,
                    reason="Generated feature is constant and contains no variation.",
                )
            )
            generated_names = [n for n in generated_names if n != name]

    # --- High missingness generated features ---
    for name in list(generated_names):
        if name not in working.columns or name in original_set:
            continue
        miss_pct = _series_stats_missing_pct(working[name])
        if miss_pct > GENERATED_FEATURE_MAX_MISSING_PERCENT:
            working = working.drop(columns=[name])
            meta = meta_by_name.get(name)
            removed.append(
                RemovedFeatureMeta(
                    feature=name,
                    source=meta.source if meta else None,
                    category=meta.category if meta else None,
                    reason=(
                        f"Generated feature missingness {miss_pct:.1f}% exceeds "
                        f"{GENERATED_FEATURE_MAX_MISSING_PERCENT}% threshold."
                    ),
                )
            )
            generated_names = [n for n in generated_names if n != name]

    # --- Exact duplicate of an original (or earlier kept) column ---
    fingerprints: dict[str, str] = {}
    for col in original_columns:
        if col in working.columns:
            fingerprints[column_value_fingerprint(working[col])] = col

    for name in list(generated_names):
        if name not in working.columns or name in original_set:
            continue
        fp = column_value_fingerprint(working[name])
        if fp in fingerprints:
            original = fingerprints[fp]
            working = working.drop(columns=[name])
            meta = meta_by_name.get(name)
            removed.append(
                RemovedFeatureMeta(
                    feature=name,
                    source=meta.source if meta else None,
                    category=meta.category if meta else None,
                    reason=f"Exact duplicate of '{original}'. Original preserved.",
                )
            )
            generated_names = [n for n in generated_names if n != name]
        else:
            fingerprints[fp] = name

    # --- Range validation (flag only) ---
    for name in generated_names:
        if name not in working.columns:
            continue
        series = working[name]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        valid = series.dropna()
        if valid.empty:
            continue
        if _MONTH_RE.search(name):
            bad = int(((valid < 1) | (valid > 12)).sum())
            if bad:
                issues.append(
                    DatetimeIssueFlag(
                        issue="range_violation",
                        columns=[name],
                        count=bad,
                        message=f"{bad} value(s) in '{name}' outside Month range 1–12.",
                    )
                )
        if _DOW_RE.search(name):
            bad = int(((valid < 0) | (valid > 6)).sum())
            if bad:
                issues.append(
                    DatetimeIssueFlag(
                        issue="range_violation",
                        columns=[name],
                        count=bad,
                        message=f"{bad} value(s) in '{name}' outside DayOfWeek range 0–6.",
                    )
                )
        if _WEEKEND_RE.search(name):
            bad = int((~valid.isin([0, 1])).sum())
            if bad:
                issues.append(
                    DatetimeIssueFlag(
                        issue="range_violation",
                        columns=[name],
                        count=bad,
                        message=f"{bad} value(s) in '{name}' are not in {{0,1}}.",
                    )
                )

    # --- Prefer nullable integers for integer-valued generated counts ---
    for name in generated_names:
        if name not in working.columns:
            continue
        series = working[name]
        if not pd.api.types.is_float_dtype(series):
            continue
        if any(
            name.endswith(suffix)
            for suffix in (
                "_Year",
                "_Month",
                "_Day",
                "_DayOfWeek",
                "_IsWeekend",
                "_Hour",
                "_Minute",
                "_CharCount",
                "_WordCount",
                "_Age_Years",
            )
        ):
            non_null = series.dropna()
            if non_null.empty:
                continue
            if np.allclose(non_null.to_numpy(dtype=float), np.round(non_null.to_numpy(dtype=float))):
                working[name] = series.round().astype("Int64")

    # Build kept metadata list
    removed_names = {r.feature for r in removed}
    for name in generated_names:
        if name in working.columns and name not in removed_names:
            meta = meta_by_name.get(name)
            if meta:
                kept_meta.append(meta)

    if len(working) != expected_rows:
        raise FeatureValidationError(
            f"Row count changed during validation: expected {expected_rows}, "
            f"got {len(working)}."
        )

    for col in original_columns:
        if col not in working.columns:
            raise FeatureValidationError(
                f"Original column '{col}' missing after validation."
            )

    return working, {
        "removed": removed,
        "issues": issues,
        "kept_generated": kept_meta,
    }
