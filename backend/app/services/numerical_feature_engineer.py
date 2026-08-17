"""Phase 5.3 — Numerical feature engineering (log, binning, meaningful ratios).

No polynomial expansion. No overwrite of original columns.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from app.schemas.feature_engineering import GeneratedFeatureMeta, SkippedFeatureMeta
from app.services.feature_engineering_config import (
    ABS_SKEWNESS_THRESHOLD,
    MIN_NON_NULL_FOR_TRANSFORM,
    MIN_UNIQUE_FOR_BINNING,
    MIN_UNIQUE_FOR_LOG,
    NUM_BINS,
)
from app.services.feature_opportunity_detector import (
    _AGE_NAME,
    _EXPERIENCE_NAME,
    _SALARY_NAME,
    detect_feature_type,
    detect_identifier,
)

_QUANTITY_NAME = re.compile(
    r"quantity|qty|units?|count|volume",
    re.IGNORECASE,
)
_REVENUE_NAME = re.compile(
    r"revenue|sales|amount|price|cost|total.?cost|charges",
    re.IGNORECASE,
)
_INCOME_NAME = re.compile(r"income|salary|wage|compensation", re.IGNORECASE)
_BIN_CANDIDATE_NAME = re.compile(
    r"(^|_)(age|income|salary|experience|tenure|years.?exp)($|_)",
    re.IGNORECASE,
)


class NumericalFeatureEngineeringError(Exception):
    """Raised for expected numerical FE failures."""


def analyze_numerical_column(series: pd.Series, column_name: str) -> dict[str, Any]:
    """Profile a numerical column without modifying it."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    n_rows = len(series)
    if non_null.empty:
        return {
            "column": column_name,
            "dtype": str(series.dtype),
            "unique_values": 0,
            "non_null": 0,
            "missing_percentage": 100.0,
            "integer_like": False,
            "skewness": 0.0,
            "zero_count": 0,
            "negative_count": 0,
        }

    vals = non_null.to_numpy(dtype=float)
    skew = float(pd.Series(vals).skew()) if len(vals) >= 3 else 0.0
    integer_like = bool(np.allclose(vals, np.round(vals)))
    return {
        "column": column_name,
        "dtype": str(series.dtype),
        "unique_values": int(non_null.nunique()),
        "non_null": int(len(non_null)),
        "missing_percentage": round((1 - len(non_null) / n_rows) * 100.0, 2) if n_rows else 0.0,
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "mean": float(non_null.mean()),
        "median": float(non_null.median()),
        "std": float(non_null.std(ddof=0)),
        "skewness": round(skew, 4),
        "zero_count": int((non_null == 0).sum()),
        "negative_count": int((non_null < 0).sum()),
        "integer_like": integer_like,
    }


def _has_variation(values: pd.Series) -> bool:
    return int(values.dropna().nunique()) > 1


def _try_add(
    working: pd.DataFrame,
    *,
    feature_name: str,
    values: pd.Series,
    source: str,
    transformation: str,
    reason: str,
    feature_type: str,
    generated: list[GeneratedFeatureMeta],
    skipped: list[SkippedFeatureMeta],
    before_stats: dict[str, Any] | None = None,
    after_stats: dict[str, Any] | None = None,
    selected: set[str] | None = None,
) -> bool:
    if selected is not None and feature_name not in selected:
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=source,
                reason="Not selected by user.",
                category="numerical",
            )
        )
        return False
    if feature_name in working.columns:
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=source,
                reason="Feature already exists — skipped.",
                category="numerical",
            )
        )
        return False
    if not _has_variation(values):
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=source,
                reason="Skipped because the candidate feature would be constant.",
                category="numerical",
            )
        )
        return False
    working[feature_name] = values
    generated.append(
        GeneratedFeatureMeta(
            feature=feature_name,
            source=source,
            feature_type=feature_type,
            category="numerical",
            transformation=transformation,
            reason=reason,
            rows_affected=int(values.notna().sum()),
            status="Created",
            before_stats=before_stats,
            after_stats=after_stats,
        )
    )
    return True


def _is_binary_numeric(profile: dict[str, Any]) -> bool:
    return int(profile.get("unique_values", 0)) <= 2


def maybe_create_log_feature(
    working: pd.DataFrame,
    column: str,
    profile: dict[str, Any],
    generated: list[GeneratedFeatureMeta],
    skipped: list[SkippedFeatureMeta],
    selected: set[str] | None = None,
) -> None:
    """Create <col>_Log via log1p when strongly right-skewed and non-negative."""
    feature_name = f"{column}_Log"
    skew = float(profile.get("skewness", 0.0))
    nunique = int(profile.get("unique_values", 0))
    non_null = int(profile.get("non_null", 0))

    if _is_binary_numeric(profile):
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=column,
                reason="Log transformation skipped: binary / near-binary feature.",
                category="numerical",
            )
        )
        return
    if nunique < MIN_UNIQUE_FOR_LOG or non_null < MIN_NON_NULL_FOR_TRANSFORM:
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=column,
                reason="Log transformation skipped: insufficient unique / non-null values.",
                category="numerical",
            )
        )
        return
    if abs(skew) < ABS_SKEWNESS_THRESHOLD:
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=column,
                reason=(
                    f"Log transformation skipped: skewness={skew:.2f} "
                    f"below threshold {ABS_SKEWNESS_THRESHOLD}."
                ),
                category="numerical",
            )
        )
        return
    if skew < 0:
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=column,
                reason="Log transformation skipped: left-skewed (not right-skewed).",
                category="numerical",
            )
        )
        return
    if int(profile.get("negative_count", 0)) > 0:
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=column,
                reason="Log transformation skipped because negative values exist.",
                category="numerical",
            )
        )
        return

    numeric = pd.to_numeric(working[column], errors="coerce")
    transformed = np.log1p(numeric)
    after_skew = float(pd.Series(transformed.dropna()).skew()) if transformed.notna().sum() >= 3 else 0.0
    before_stats = {"skewness": skew, "min": profile.get("min"), "max": profile.get("max")}
    after_stats = {
        "skewness": round(after_skew, 4),
        "min": float(transformed.min()) if transformed.notna().any() else None,
        "max": float(transformed.max()) if transformed.notna().any() else None,
    }
    _try_add(
        working,
        feature_name=feature_name,
        values=transformed.astype(float),
        source=column,
        transformation="log1p",
        reason=(
            f"{column} was strongly right-skewed (skewness={skew:.2f}). "
            f"log1p reduced skewness to {after_skew:.2f}."
        ),
        feature_type="Float",
        generated=generated,
        skipped=skipped,
        before_stats=before_stats,
        after_stats=after_stats,
        selected=selected,
    )


def maybe_create_binned_feature(
    working: pd.DataFrame,
    column: str,
    profile: dict[str, Any],
    generated: list[GeneratedFeatureMeta],
    skipped: list[SkippedFeatureMeta],
    selected: set[str] | None = None,
) -> None:
    """Quantile binning for interpretable continuous variables (e.g. Age)."""
    feature_name = f"{column}_Binned"
    if not _BIN_CANDIDATE_NAME.search(column) and not _AGE_NAME.search(column):
        return  # Only consider semantically bin-friendly names

    nunique = int(profile.get("unique_values", 0))
    non_null = int(profile.get("non_null", 0))
    if _is_binary_numeric(profile) or nunique < MIN_UNIQUE_FOR_BINNING:
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=column,
                reason="Binning skipped: insufficient unique values / binary feature.",
                category="numerical",
            )
        )
        return
    if non_null < MIN_NON_NULL_FOR_TRANSFORM:
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=column,
                reason="Binning skipped: insufficient non-null observations.",
                category="numerical",
            )
        )
        return

    numeric = pd.to_numeric(working[column], errors="coerce")
    try:
        binned = pd.qcut(
            numeric,
            q=NUM_BINS,
            duplicates="drop",
        )
    except (ValueError, TypeError):
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=column,
                reason="Binning skipped: could not form stable quantile bins.",
                category="numerical",
            )
        )
        return

    # Readable interval labels
    labels = binned.astype(str)
    labels = labels.where(numeric.notna(), other=pd.NA)
    _try_add(
        working,
        feature_name=feature_name,
        values=labels,
        source=column,
        transformation=f"Quantile binning (q={NUM_BINS})",
        reason=(
            f"{column} has a meaningful continuous range "
            f"({profile.get('min')}–{profile.get('max')}) with {nunique} unique values; "
            "quantile bins aid interpretability."
        ),
        feature_type="Categorical",
        generated=generated,
        skipped=skipped,
        selected=selected,
    )


def _ratio_pairs(columns: list[str], type_map: dict[str, str]) -> list[tuple[str, str, str]]:
    """Meaningful numerator/denominator pairs only (no all-pairs explosion)."""
    numeric_cols = [c for c in columns if type_map.get(c) == "numerical"]
    pairs: list[tuple[str, str, str]] = []

    numerators = [
        c
        for c in numeric_cols
        if _INCOME_NAME.search(c) or _SALARY_NAME.search(c) or _REVENUE_NAME.search(c)
    ]
    experience = [c for c in numeric_cols if _EXPERIENCE_NAME.search(c)]
    quantities = [c for c in numeric_cols if _QUANTITY_NAME.search(c)]

    for num in numerators:
        for exp in experience:
            if num != exp:
                pairs.append((num, exp, f"{num}_Per_{exp}"))
        for qty in quantities:
            if num != qty:
                # Prefer readable names for revenue/quantity
                if _REVENUE_NAME.search(num) and _QUANTITY_NAME.search(qty):
                    name = f"{num}_Per_Unit" if "unit" in qty.lower() or "qty" in qty.lower() else f"{num}_Per_{qty}"
                else:
                    name = f"{num}_Per_{qty}"
                pairs.append((num, qty, name))

    # Deduplicate by feature name
    seen: set[str] = set()
    unique: list[tuple[str, str, str]] = []
    for item in pairs:
        if item[2] not in seen:
            seen.add(item[2])
            unique.append(item)
    return unique


def maybe_create_ratio_features(
    working: pd.DataFrame,
    type_map: dict[str, str],
    generated: list[GeneratedFeatureMeta],
    skipped: list[SkippedFeatureMeta],
    issues: list[dict[str, Any]],
    selected: set[str] | None = None,
) -> None:
    cols = [str(c) for c in working.columns]
    for num, den, feature_name in _ratio_pairs(cols, type_map):
        feature_name = re.sub(r"[^A-Za-z0-9_]+", "_", feature_name)
        num_s = pd.to_numeric(working[num], errors="coerce")
        den_s = pd.to_numeric(working[den], errors="coerce")
        zero_mask = den_s.eq(0)
        zero_count = int(zero_mask.fillna(False).sum())
        ratio = num_s / den_s.replace(0, np.nan)
        # Replace ±inf just in case
        inf_count = int(np.isinf(ratio.to_numpy(dtype=float, na_value=np.nan)).sum())
        if inf_count:
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            issues.append(
                {
                    "issue": "infinite_ratio",
                    "columns": [num, den],
                    "count": inf_count,
                    "message": (
                        f"{inf_count} infinite value(s) in {feature_name} replaced with NaN."
                    ),
                }
            )
        if zero_count:
            issues.append(
                {
                    "issue": "division_by_zero",
                    "columns": [num, den],
                    "count": zero_count,
                    "message": (
                        f"{zero_count} row(s) had {den}=0; ratio set to NaN for those rows."
                    ),
                }
            )
        _try_add(
            working,
            feature_name=feature_name,
            values=ratio.astype(float),
            source=f"{num}, {den}",
            transformation=f"{num} / {den} (safe division)",
            reason=(
                f"Meaningful relationship between '{num}' and '{den}' "
                "suggested a per-unit / per-experience style ratio."
            ),
            feature_type="Float",
            generated=generated,
            skipped=skipped,
            selected=selected,
        )


def engineer_numerical_features(
    df: pd.DataFrame,
    *,
    selected: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply Phase 5.3 numerical FE on a copy of ``df``."""
    working = df.copy()
    before_rows = len(working)
    generated: list[GeneratedFeatureMeta] = []
    skipped: list[SkippedFeatureMeta] = []
    issues: list[dict[str, Any]] = []
    polynomial_recommendations: list[str] = []
    profiles: list[dict[str, Any]] = []

    type_map: dict[str, str] = {}
    for column in list(working.columns):
        col = str(column)
        ftype, _ = detect_feature_type(working[col], col)
        type_map[col] = ftype

    for column in list(working.columns):
        col = str(column)
        if type_map.get(col) != "numerical":
            continue
        if detect_identifier(working[col], col):
            skipped.append(
                SkippedFeatureMeta(
                    feature=f"{col}_Log",
                    source=col,
                    reason="Skipped: column appears to be an identifier.",
                    category="numerical",
                )
            )
            continue

        profile = analyze_numerical_column(working[col], col)
        profiles.append(profile)
        maybe_create_log_feature(working, col, profile, generated, skipped, selected=selected)
        maybe_create_binned_feature(working, col, profile, generated, skipped, selected=selected)

        # Polynomial relationships are recommendations only
        if _AGE_NAME.search(col) or _SALARY_NAME.search(col):
            polynomial_recommendations.append(
                f"Polynomial expansion of '{col}' was considered but not applied "
                "(avoids feature explosion)."
            )

    maybe_create_ratio_features(
        working, type_map, generated, skipped, issues, selected=selected
    )

    if len(working) != before_rows:
        raise NumericalFeatureEngineeringError(
            "Row count changed during numerical feature engineering."
        )

    return working, {
        "generated": generated,
        "skipped": skipped,
        "issues": issues,
        "profiles": profiles,
        "polynomial_recommendations": list(dict.fromkeys(polynomial_recommendations)),
    }
