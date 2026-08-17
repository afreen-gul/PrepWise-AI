"""Intelligent data cleaning (Phase 4).

Creates a cleaned *copy* of an uploaded dataset. The original file under
``uploads/`` is never modified. All transformations are logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset
from app.schemas.cleaning import (
    CleaningApplyResponse,
    CleaningConfig,
    CleaningLogEntry,
    CleaningPreviewResponse,
    CleaningSummary,
    DatasetSnapshot,
    IssueTransition,
    OutlierStrategy,
)
from app.services.dataset_profiler import _is_datetime_like, _name_matches_identifier
from app.services.dataset_service import DatasetServiceError, read_dataset_csv
from app.services.duplicate_columns import (
    find_exact_duplicate_column_groups,
    process_duplicate_columns,
)

# Safe conversion thresholds
NUMERIC_CONVERT_RATIO = 0.95
DATETIME_CONVERT_RATIO = 0.90
SKEW_MEDIAN_THRESHOLD = 1.0
MIN_ROWS_FOR_SKEW = 8
MIN_ROWS_FOR_IQR = 8
PREVIEW_ROWS = 10

# Categorical imputation confidence thresholds (non-null distribution).
CATEGORICAL_DOMINANT_MIN_SHARE = 0.45
CATEGORICAL_MIN_GAP_TO_SECOND = 0.10
CATEGORICAL_CAUTIOUS_MISSING_RATIO = 0.30
CATEGORICAL_MIN_NON_NULL_FOR_MODE = 3
CATEGORICAL_UNKNOWN_LABEL = "Unknown"

_AGE_NAME = re.compile(r"(^|_)age($|_)", re.IGNORECASE)
_SALARY_NAME = re.compile(r"salary|wage|income|compensation", re.IGNORECASE)
_QUANTITY_NAME = re.compile(r"quantity|qty|count|amount", re.IGNORECASE)

# Domain caps for obviously impossible ages (not statistical outliers).
MAX_PLAUSIBLE_AGE = 120
MAX_TRANSFORM_SAMPLES = 8


@dataclass(frozen=True)
class CategoricalImputationParams:
    """Configurable thresholds for categorical missing-value imputation."""

    min_group_size: int = 10
    min_group_confidence: float = 0.60
    high_confidence_threshold: float = 0.75
    max_grouping_cardinality: int = 50
    global_mode_min_confidence: float = 0.60
    min_valid_observations: int = 20
    min_unique_categories: int = 2
    random_state: int = 42

    @classmethod
    def from_cleaning_config(cls, config: CleaningConfig | None) -> CategoricalImputationParams:
        if config is None:
            return cls()
        return cls(
            min_group_size=config.min_group_size,
            min_group_confidence=config.min_group_confidence,
            high_confidence_threshold=config.high_confidence_threshold,
            max_grouping_cardinality=config.max_grouping_cardinality,
            global_mode_min_confidence=config.global_mode_min_confidence,
            min_valid_observations=config.min_valid_observations,
            min_unique_categories=config.min_unique_categories,
            random_state=config.categorical_random_state,
        )


def _confidence_tier(ratio: float, params: CategoricalImputationParams) -> str:
    if ratio >= params.high_confidence_threshold:
        return "high"
    if ratio >= params.min_group_confidence:
        return "medium"
    return "low"


def _is_categorical_series(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    return True


def analyze_categorical_distribution(
    series: pd.Series,
    *,
    missing_count: int,
    missing_ratio: float,
) -> dict[str, Any]:
    """Step 1 — missing stats plus frequency distribution."""
    profile = _categorical_frequency_profile(series)
    profile["missing_count"] = missing_count
    profile["missing_percentage"] = round(missing_ratio * 100.0, 2)
    profile["missing_ratio"] = round(missing_ratio, 4)
    return profile


def is_approximately_uniform(profile: dict[str, Any]) -> bool:
    """True when no category has clear dominance (do not force balance)."""
    if profile["n_categories"] < 2:
        return False
    return (
        profile["dominant_share"] < CATEGORICAL_DOMINANT_MIN_SHARE
        and (profile["dominant_share"] - profile["second_share"])
        <= CATEGORICAL_MIN_GAP_TO_SECOND
    )


def calculate_group_confidence(valid_target: pd.Series) -> tuple[float, Any, int]:
    """Dominant share among valid target values in a group."""
    counts = valid_target.value_counts(dropna=True)
    total = int(counts.sum())
    if total == 0:
        return 0.0, None, 0
    top_label = counts.index[0]
    top_count = int(counts.iloc[0])
    return top_count / total, top_label, total


def find_grouping_candidates(
    df: pd.DataFrame,
    target_column: str,
    params: CategoricalImputationParams,
    duplicate_of: dict[str, str] | None = None,
    log: list[CleaningLogEntry] | None = None,
) -> list[str]:
    """Categorical/low-cardinality features suitable for group-based inference."""
    duplicate_of = duplicate_of or {}
    n_rows = len(df)
    scored: list[tuple[str, int]] = []

    for column in df.columns:
        col_name = str(column)
        if col_name == target_column:
            continue
        if col_name in duplicate_of:
            if log is not None:
                _log(
                    log,
                    "duplicate_column_grouping_excluded",
                    column=col_name,
                    duplicate_of=duplicate_of[col_name],
                    message=(
                        f"'{col_name}' is an exact duplicate of "
                        f"'{duplicate_of[col_name]}' — not used as an independent "
                        "grouping feature."
                    ),
                )
            continue
        series = df[column]
        if not _is_categorical_series(series):
            continue
        if _name_matches_identifier(col_name):
            continue
        nunique = int(series.nunique(dropna=True))
        if nunique <= 1:
            continue
        if nunique > params.max_grouping_cardinality:
            continue
        if n_rows > 0 and (nunique / n_rows) > 0.5:
            continue
        scored.append((col_name, nunique))

    scored.sort(key=lambda item: item[1])
    return [name for name, _ in scored]


def group_based_mode_imputation(
    df: pd.DataFrame,
    row_index: Any,
    target_column: str,
    grouping_column: str,
    params: CategoricalImputationParams,
) -> dict[str, Any] | None:
    """Infer one missing target from the dominant category within a group."""
    group_value = df.loc[row_index, grouping_column]
    if pd.isna(group_value):
        return None

    in_group = df[grouping_column] == group_value
    valid = df.loc[in_group, target_column].dropna()
    if len(valid) < params.min_group_size:
        return None

    confidence_ratio, selected, group_size = calculate_group_confidence(valid)
    if confidence_ratio < params.min_group_confidence:
        return None

    tier = _confidence_tier(confidence_ratio, params)
    return {
        "method": "group_based_mode",
        "method_label": "Group-Based Mode",
        "fill_value": selected,
        "selected_category": _jsonable(selected),
        "grouping_feature": grouping_column,
        "group": _jsonable(group_value),
        "group_size": group_size,
        "confidence_ratio": round(confidence_ratio, 4),
        "confidence_pct": round(confidence_ratio * 100.0, 2),
        "confidence": tier,
        "reason": (
            f"{selected!r} is dominant among valid {target_column} values "
            f"for {grouping_column}={group_value!r} "
            f"({confidence_ratio:.0%} of {group_size} observations)."
        ),
    }


def pick_best_group_imputation(
    df: pd.DataFrame,
    row_index: Any,
    target_column: str,
    grouping_columns: list[str],
    params: CategoricalImputationParams,
) -> dict[str, Any] | None:
    """Try grouping features; return the strongest group-level inference."""
    best: dict[str, Any] | None = None
    best_conf = -1.0
    for group_col in grouping_columns:
        result = group_based_mode_imputation(
            df, row_index, target_column, group_col, params
        )
        if result is None:
            continue
        conf = float(result["confidence_ratio"])
        if conf > best_conf:
            best_conf = conf
            best = result
    return best


def assess_global_mode(
    profile: dict[str, Any],
    params: CategoricalImputationParams,
) -> dict[str, Any]:
    """Step 2 — global dominant category if share meets threshold."""
    dominant = profile["dominant_category"]
    dominant_pct = profile["dominant_percentage"]
    share = profile["dominant_share"]
    threshold_pct = params.global_mode_min_confidence * 100.0

    if profile["non_null_count"] < CATEGORICAL_MIN_NON_NULL_FOR_MODE:
        return {
            "selected": False,
            "reason": (
                f"Not used — only {profile['non_null_count']} valid observation(s)."
            ),
        }

    if profile["non_null_count"] < params.min_valid_observations:
        return {
            "selected": False,
            "reason": (
                f"Not used — only {profile['non_null_count']} valid observations "
                f"(minimum {params.min_valid_observations})."
            ),
        }

    if dominant is None:
        return {"selected": False, "reason": "Not used — no dominant category."}

    if share >= params.global_mode_min_confidence:
        return {
            "selected": True,
            "fill_value": dominant,
            "confidence": _confidence_tier(share, params),
            "confidence_pct": round(dominant_pct, 2),
            "reason": (
                f"Selected — {dominant!r} represented {dominant_pct:.1f}% of valid "
                f"observations (≥ {threshold_pct:.0f}% threshold)."
            ),
        }

    return {
        "selected": False,
        "reason": (
            f"Not used — dominant category = {dominant_pct:.1f}%, "
            f"below {threshold_pct:.0f}% threshold."
        ),
    }


def can_use_distribution_imputation(
    profile: dict[str, Any],
    params: CategoricalImputationParams,
) -> dict[str, Any]:
    """Step 3 — empirical distribution sampling eligibility."""
    non_null = profile["non_null_count"]
    n_cats = profile["n_categories"]

    if non_null < params.min_valid_observations:
        return {
            "selected": False,
            "reason": (
                f"Not used — {non_null} valid observations "
                f"(minimum {params.min_valid_observations})."
            ),
        }
    if n_cats < params.min_unique_categories:
        return {
            "selected": False,
            "reason": (
                f"Not used — only {n_cats} categor"
                f"{'y' if n_cats == 1 else 'ies'}."
            ),
        }
    return {
        "selected": True,
        "reason": (
            f"Selected — {non_null} valid observations and {n_cats} categories; "
            "replacements sampled from observed proportions (not balanced)."
        ),
    }


def sample_from_empirical_distribution(
    valid: pd.Series,
    n_missing: int,
    random_state: int,
) -> tuple[list[Any], dict[str, int]]:
    """Sample missing fills from observed category proportions."""
    if n_missing <= 0:
        return [], {}
    counts = valid.value_counts(normalize=True, dropna=True)
    categories = counts.index.tolist()
    probs = counts.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    drawn = rng.choice(categories, size=n_missing, replace=True, p=probs)
    replacement_counts: dict[str, int] = {}
    values: list[Any] = []
    for value in drawn:
        values.append(value)
        key = str(_jsonable(value))
        replacement_counts[key] = replacement_counts.get(key, 0) + 1
    return values, replacement_counts


def evaluate_insufficient_evidence(
    profile: dict[str, Any],
    params: CategoricalImputationParams,
) -> dict[str, Any]:
    """Step 4 — Unknown or flag when no method applies."""
    non_null = profile["non_null_count"]
    if non_null < CATEGORICAL_MIN_NON_NULL_FOR_MODE:
        return {
            "action": "review",
            "method": "flag_for_review",
            "method_label": "Flag for Review",
            "fill_value": None,
            "confidence": "none",
            "reason": "Insufficient valid observations for reliable inference.",
        }
    return {
        "action": "fill",
        "method": "unknown_category",
        "method_label": "Unknown",
        "fill_value": CATEGORICAL_UNKNOWN_LABEL,
        "confidence": "low",
        "reason": "Insufficient evidence for reliable categorical imputation.",
    }


def build_decision_process(
    *,
    grouping_columns: list[str],
    group_used_count: int,
    global_assessment: dict[str, Any],
    distribution_assessment: dict[str, Any],
    final_method: str,
) -> list[dict[str, Any]]:
    """Explainable step-by-step decision trail for the UI."""
    group_reason = (
        f"Used for {group_used_count} missing value(s)."
        if group_used_count
        else (
            "Not used — no reliable grouping relationship found "
            f"({len(grouping_columns)} candidate feature(s) checked)."
        )
    )
    return [
        {
            "step": "Group-based inference",
            "status": "selected" if group_used_count else "not_used",
            "detail": group_reason,
        },
        {
            "step": "Global mode",
            "status": "selected" if final_method == "global_mode" else "not_used",
            "detail": global_assessment.get("reason", ""),
        },
        {
            "step": "Distribution-based",
            "status": "selected" if final_method == "distribution_based" else "not_used",
            "detail": distribution_assessment.get("reason", ""),
        },
        {
            "step": "Unknown / review",
            "status": "selected"
            if final_method in {"unknown_category", "flag_for_review"}
            else "not_used",
            "detail": (
                "Selected — insufficient evidence for reliable categorical imputation."
                if final_method in {"unknown_category", "flag_for_review"}
                else "Not used — a stronger method was applied."
            ),
        },
    ]


def _format_distribution_replacement(replacement_counts: dict[str, int]) -> str:
    if not replacement_counts:
        return "—"
    if len(replacement_counts) == 1:
        return next(iter(replacement_counts))
    return "/".join(sorted(replacement_counts.keys()))


def log_categorical_imputation(
    log: list[CleaningLogEntry],
    *,
    column: str,
    missing_before: int,
    profile: dict[str, Any],
    summary_table: list[dict[str, Any]],
    row_details: list[dict[str, Any]],
    from_invalid_count: int,
    before_dtype: str,
    final_dtype: str,
    before_series: pd.Series,
    after_series: pd.Series,
    missing_mask: pd.Series,
) -> None:
    """Record categorical imputation with UI-friendly summary."""
    primary = summary_table[0] if summary_table else {}
    samples = _sample_pairs(before_series, after_series, missing_mask)
    after_missing = int(after_series.isna().sum())

    _log(
        log,
        "missing_value_imputation",
        column=column,
        method=primary.get("method_key", primary.get("method", "")),
        method_label=primary.get("method"),
        fill_value=primary.get("replacement"),
        selected_category=primary.get("replacement"),
        before_missing=missing_before,
        after_missing=after_missing,
        from_invalid_domain_count=from_invalid_count,
        before_dtype=before_dtype,
        final_dtype=final_dtype,
        dominant_category=profile.get("dominant_category"),
        dominant_percentage=profile.get("dominant_percentage"),
        frequency_distribution=profile.get("distribution"),
        missing_count=profile.get("missing_count"),
        missing_percentage=profile.get("missing_percentage"),
        n_categories=profile.get("n_categories"),
        confidence=primary.get("confidence"),
        confidence_pct=primary.get("confidence_pct"),
        reason=primary.get("reason"),
        grouping_feature=primary.get("grouping_feature"),
        group=primary.get("group"),
        group_size=primary.get("group_size"),
        imputation_summary_table=summary_table,
        imputation_row_details=row_details[:MAX_TRANSFORM_SAMPLES],
        decision_process=primary.get("decision_process"),
        replacement_counts=primary.get("replacement_counts"),
        transformations=samples,
        message=primary.get("reason", f"Categorical imputation on '{column}'."),
    )


def impute_categorical_column(
    df: pd.DataFrame,
    column: str,
    log: list[CleaningLogEntry],
    params: CategoricalImputationParams,
    *,
    missing_before: int,
    missing_ratio: float,
    from_invalid_count: int,
) -> pd.Series:
    """Group-based → global mode → distribution-based → unknown/review."""
    series = df[column]
    before_series = series.copy()
    missing_mask = series.isna()
    if not missing_mask.any():
        return series

    profile = analyze_categorical_distribution(
        series,
        missing_count=missing_before,
        missing_ratio=missing_ratio,
    )

    duplicate_of = find_exact_duplicate_column_groups(df)
    grouping_columns = find_grouping_candidates(
        df, column, params, duplicate_of=duplicate_of, log=log
    )
    global_assessment = assess_global_mode(profile, params)
    distribution_assessment = can_use_distribution_imputation(profile, params)

    filled = series.copy()
    row_details: list[dict[str, Any]] = []
    group_used_count = 0
    remaining_indices: list[Any] = []

    for row_index in series.index[missing_mask]:
        group_result = pick_best_group_imputation(
            df, row_index, column, grouping_columns, params
        )
        if group_result is not None:
            filled.loc[row_index] = group_result["fill_value"]
            row_details.append({**group_result, "row_index": _jsonable(row_index)})
            group_used_count += 1
        else:
            remaining_indices.append(row_index)

    replacement_counts: dict[str, int] | None = None
    bulk_method = "group_based_mode"
    bulk_reason = ""
    insufficient: dict[str, Any] | None = None

    if remaining_indices:
        if global_assessment.get("selected"):
            bulk_method = "global_mode"
            dominant = global_assessment["fill_value"]
            dominant_pct = global_assessment.get("confidence_pct", profile["dominant_percentage"])
            bulk_reason = (
                f"{dominant!r} represented {dominant_pct:.1f}% of valid observations."
            )
            for row_index in remaining_indices:
                filled.loc[row_index] = dominant
                row_details.append(
                    {
                        "method": "global_mode",
                        "method_label": "Global Mode",
                        "fill_value": dominant,
                        "selected_category": _jsonable(dominant),
                        "confidence": global_assessment.get("confidence", "high"),
                        "confidence_pct": dominant_pct,
                        "reason": bulk_reason,
                        "row_index": _jsonable(row_index),
                    }
                )
        elif distribution_assessment.get("selected"):
            bulk_method = "distribution_based"
            valid = series.dropna()
            sampled_values, replacement_counts = sample_from_empirical_distribution(
                valid,
                len(remaining_indices),
                params.random_state,
            )
            replacement_label = _format_distribution_replacement(replacement_counts)
            bulk_reason = (
                "Global mode was below "
                f"{params.global_mode_min_confidence:.0%}%, so replacements were "
                "sampled according to the observed distribution."
            )
            for row_index, fill_value in zip(remaining_indices, sampled_values, strict=True):
                filled.loc[row_index] = fill_value
                row_details.append(
                    {
                        "method": "distribution_based",
                        "method_label": "Distribution-Based Imputation",
                        "fill_value": fill_value,
                        "selected_category": _jsonable(fill_value),
                        "confidence": "medium",
                        "confidence_pct": None,
                        "reason": bulk_reason,
                        "replacement_counts": replacement_counts,
                        "row_index": _jsonable(row_index),
                    }
                )
        else:
            insufficient = evaluate_insufficient_evidence(profile, params)
            bulk_method = str(insufficient["method"])
            bulk_reason = str(insufficient["reason"])
            if insufficient["action"] == "review":
                _log(
                    log,
                    "categorical_imputation_review",
                    column=column,
                    method=insufficient["method"],
                    confidence=insufficient["confidence"],
                    reason=insufficient["reason"],
                    frequency_distribution=profile.get("distribution"),
                    before_missing=missing_before,
                    after_missing=int(filled.isna().sum()),
                    missing_ratio=missing_ratio,
                    message=insufficient["reason"],
                )
                return series
            for row_index in remaining_indices:
                filled.loc[row_index] = insufficient["fill_value"]
                row_details.append(
                    {
                        **insufficient,
                        "method_label": insufficient.get("method_label", "Unknown"),
                        "row_index": _jsonable(row_index),
                    }
                )

    if not remaining_indices and group_used_count:
        bulk_method = "group_based_mode"

    decision_process = build_decision_process(
        grouping_columns=grouping_columns,
        group_used_count=group_used_count,
        global_assessment=global_assessment,
        distribution_assessment=distribution_assessment,
        final_method=bulk_method,
    )

    summary_table = _build_categorical_summary_table(
        column=column,
        missing_before=missing_before,
        row_details=row_details,
        decision_process=decision_process,
        replacement_counts=replacement_counts,
    )

    log_categorical_imputation(
        log,
        column=column,
        missing_before=missing_before,
        profile=profile,
        summary_table=summary_table,
        row_details=row_details,
        from_invalid_count=from_invalid_count,
        before_dtype=str(series.dtype),
        final_dtype=str(filled.dtype),
        before_series=before_series,
        after_series=filled,
        missing_mask=missing_mask,
    )
    return filled


def _build_categorical_summary_table(
    *,
    column: str,
    missing_before: int,
    row_details: list[dict[str, Any]],
    decision_process: list[dict[str, Any]] | None = None,
    replacement_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate row-level decisions for the UI table (one or more rows)."""
    if not row_details:
        return []

    by_method: dict[str, list[dict[str, Any]]] = {}
    for detail in row_details:
        key = str(detail.get("method", "unknown"))
        by_method.setdefault(key, []).append(detail)

    rows: list[dict[str, Any]] = []
    for method_key, details in by_method.items():
        sample = details[0]
        label = sample.get("method_label") or method_key
        replacement = _jsonable(sample.get("fill_value"))
        conf = sample.get("confidence", "low")
        conf_pct = sample.get("confidence_pct")
        if conf_pct is None and sample.get("dominant_frequency_pct") is not None:
            conf_pct = sample["dominant_frequency_pct"]

        reason = sample.get("reason", "")
        if method_key == "group_based_mode":
            reason = (
                f"Group-based mode using {sample.get('grouping_feature')!r} "
                f"({len(details)} missing value(s); "
                f"example group {sample.get('group')!r})."
            )
        elif method_key == "distribution_based":
            counts = replacement_counts or sample.get("replacement_counts") or {}
            replacement = _format_distribution_replacement(
                {str(k): int(v) for k, v in counts.items()}
            )
            if counts:
                parts = [f"{k}: {v}" for k, v in sorted(counts.items())]
                reason = f"{reason} Counts: {', '.join(parts)}."

        rows.append(
            {
                "column": column,
                "missing": len(details),
                "method": label,
                "method_key": method_key,
                "replacement": replacement,
                "confidence": conf,
                "confidence_pct": conf_pct,
                "reason": reason,
                "grouping_feature": sample.get("grouping_feature"),
                "group": sample.get("group"),
                "group_size": sample.get("group_size"),
                "replacement_counts": replacement_counts
                if method_key == "distribution_based"
                else None,
                "decision_process": decision_process,
            }
        )

    if len(rows) == 1:
        rows[0]["missing"] = missing_before
    elif rows:
        rows[0]["decision_process"] = decision_process
    return rows


class DataCleanerError(Exception):
    """Raised for expected, user-facing cleaning failures."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(
    entries: list[CleaningLogEntry],
    operation: str,
    *,
    column: str | None = None,
    message: str = "",
    **details: Any,
) -> None:
    entries.append(
        CleaningLogEntry(
            operation=operation,
            column=column,
            details=details,
            message=message,
        )
    )


def _jsonable(value: Any) -> Any:
    """Convert a scalar to a JSON-friendly representation."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _sample_pairs(
    before_series: pd.Series,
    after_series: pd.Series,
    mask: pd.Series,
    *,
    limit: int = MAX_TRANSFORM_SAMPLES,
) -> list[dict[str, Any]]:
    """Build before/after samples for transformed rows."""
    indices = before_series.index[mask.fillna(False)][:limit]
    pairs: list[dict[str, Any]] = []
    for idx in indices:
        pairs.append(
            {
                "row_index": _jsonable(idx),
                "before": _jsonable(before_series.loc[idx]),
                "after": _jsonable(after_series.loc[idx]),
            }
        )
    return pairs


def _total_missing(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())


def _duplicate_row_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df.duplicated().sum())


def _is_skewed(series: pd.Series) -> bool:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < MIN_ROWS_FOR_SKEW:
        return False
    try:
        return abs(float(clean.skew())) >= SKEW_MEDIAN_THRESHOLD
    except Exception:
        return False


def _is_integer_like(series: pd.Series) -> bool:
    """Return True when non-null values are integer-valued (not continuous).

    True for integer dtypes and for floats whose observed values are all whole
    numbers (e.g. Age stored as float64 after CSV load). False when any
    fractional value is present, so continuous features keep decimals.
    """
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_integer_dtype(series):
        return True
    if not pd.api.types.is_numeric_dtype(series):
        return False

    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return False
    # Allow tiny float noise from prior arithmetic, but reject true fractions.
    return bool((clean - clean.round()).abs().le(1e-9).all())


def _restore_integer_dtype(series: pd.Series, fill_value: float) -> tuple[pd.Series, int]:
    """Round imputed series to whole numbers and cast to nullable Int64."""
    rounded_fill = int(round(float(fill_value)))
    restored = series.round().astype("Int64")
    return restored, rounded_fill


def _categorical_frequency_profile(series: pd.Series) -> dict[str, Any]:
    """Build frequency distribution stats for a categorical series."""
    non_null = series.dropna()
    counts = non_null.value_counts(dropna=True)
    total = int(counts.sum())
    distribution: list[dict[str, Any]] = []
    for label, count in counts.items():
        distribution.append(
            {
                "category": _jsonable(label),
                "count": int(count),
                "percentage": round((int(count) / total) * 100.0, 2) if total else 0.0,
            }
        )
    top = distribution[0] if distribution else None
    second = distribution[1] if len(distribution) > 1 else None
    dominant_share = (top["percentage"] / 100.0) if top else 0.0
    second_share = (second["percentage"] / 100.0) if second else 0.0
    return {
        "non_null_count": total,
        "n_categories": len(distribution),
        "distribution": distribution[:12],  # keep log payloads bounded
        "dominant_category": top["category"] if top else None,
        "dominant_percentage": top["percentage"] if top else 0.0,
        "second_category": second["category"] if second else None,
        "second_percentage": second["percentage"] if second else 0.0,
        "gap_to_second": round((dominant_share - second_share) * 100.0, 2),
        "dominant_share": dominant_share,
        "second_share": second_share,
    }


def _iqr_bounds(series: pd.Series) -> tuple[float, float] | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < MIN_ROWS_FOR_IQR:
        return None
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return None
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _outlier_mask(series: pd.Series) -> pd.Series:
    bounds = _iqr_bounds(series)
    if bounds is None:
        return pd.Series(False, index=series.index)
    lower, upper = bounds
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric < lower) | (numeric > upper)


def _count_outliers(df: pd.DataFrame) -> int:
    total = 0
    for column in df.columns:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        total += int(_outlier_mask(series).fillna(False).sum())
    return total


def _constant_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(col)
        for col in df.columns
        if df[col].nunique(dropna=True) <= 1
    ]


def _safe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure unique column labels without changing semantics."""
    if df.columns.is_unique:
        return df
    seen: dict[str, int] = {}
    new_names: list[str] = []
    for name in df.columns.astype(str):
        count = seen.get(name, 0)
        if count == 0:
            new_names.append(name)
        else:
            new_names.append(f"{name}__dup{count}")
        seen[name] = count + 1
    out = df.copy()
    out.columns = new_names
    return out


def _snapshot(df: pd.DataFrame, *, columns_modified: list[str] | None = None) -> DatasetSnapshot:
    return DatasetSnapshot(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        missing_values=_total_missing(df),
        duplicate_rows=_duplicate_row_count(df),
        outliers_detected=_count_outliers(df),
        columns_modified=columns_modified or [],
    )


def _build_preview_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    head = df.head(PREVIEW_ROWS).where(pd.notnull(df.head(PREVIEW_ROWS)), None)
    return head.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------


def normalize_empty_strings(
    df: pd.DataFrame,
    log: list[CleaningLogEntry],
) -> pd.DataFrame:
    """Convert empty / whitespace-only strings to NaN."""
    out = df.copy()
    for column in out.columns:
        series = out[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue
        try:
            stripped = series.astype(str)
            mask = series.notna() & stripped.str.strip().eq("")
            # Also catch literal "nan" from prior string casts of real NaN? skip those
            count = int(mask.sum())
            if count == 0:
                continue
            out.loc[mask, column] = pd.NA
            _log(
                log,
                "empty_string_normalization",
                column=str(column),
                empty_strings_converted=count,
                message=f"Converted {count} empty/whitespace string(s) to missing.",
            )
        except Exception as exc:  # pragma: no cover - defensive
            _log(
                log,
                "empty_string_normalization_skipped",
                column=str(column),
                error=str(exc),
                message="Skipped empty-string normalization for this column.",
            )
    return out


def convert_safe_dtypes(
    df: pd.DataFrame,
    log: list[CleaningLogEntry],
) -> pd.DataFrame:
    """Convert object columns to numeric/datetime only when conversion is safe."""
    out = df.copy()
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        old_dtype = str(series.dtype)

        # Numeric first (safer / more common for CSV text numbers)
        try:
            coerced = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
            ratio = float(coerced.notna().mean())
            if ratio >= NUMERIC_CONVERT_RATIO:
                out[column] = pd.to_numeric(series, errors="coerce")
                _log(
                    log,
                    "dtype_conversion",
                    column=str(column),
                    old_dtype=old_dtype,
                    new_dtype=str(out[column].dtype),
                    conversion_ratio=round(ratio, 4),
                    message=f"Converted '{column}' from {old_dtype} to numeric.",
                )
                continue
        except Exception as exc:
            _log(
                log,
                "dtype_conversion_skipped",
                column=str(column),
                error=str(exc),
                message="Numeric conversion skipped.",
            )

        # Datetime when clearly parseable
        try:
            if _is_datetime_like(series):
                sample = non_null.head(200)
                parsed_sample = pd.to_datetime(
                    sample, errors="coerce", format="mixed"
                )
                if float(parsed_sample.notna().mean()) >= DATETIME_CONVERT_RATIO:
                    out[column] = pd.to_datetime(series, errors="coerce", format="mixed")
                    _log(
                        log,
                        "dtype_conversion",
                        column=str(column),
                        old_dtype=old_dtype,
                        new_dtype=str(out[column].dtype),
                        message=f"Converted '{column}' from {old_dtype} to datetime.",
                    )
        except Exception as exc:
            _log(
                log,
                "dtype_conversion_skipped",
                column=str(column),
                error=str(exc),
                message="Datetime conversion skipped.",
            )
    return out


def _domain_invalid_mask(column_name: str, numeric: pd.Series) -> tuple[pd.Series, str] | None:
    """Return (mask, issue_type) for domain-invalid values, or None if N/A.

    These are *not* statistical outliers — they violate basic domain rules.
    """
    if _AGE_NAME.search(column_name):
        # Impossible ages: negative or above a hard biological/admin ceiling.
        mask = (numeric < 0) | (numeric > MAX_PLAUSIBLE_AGE)
        return mask.fillna(False), "impossible_age"
    if _SALARY_NAME.search(column_name):
        return (numeric < 0).fillna(False), "negative_salary"
    if _QUANTITY_NAME.search(column_name):
        return (numeric < 0).fillna(False), "negative_quantity"
    return None


def handle_invalid_numeric_values(
    df: pd.DataFrame,
    log: list[CleaningLogEntry],
) -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    """Convert domain-invalid values to missing (not IQR outliers).

    Returns the cleaned frame and a map of column → indices converted to missing
    so imputation can attribute fill-ins that came from invalid values.
    """
    out = df.copy()
    invalid_indices: dict[str, pd.Index] = {}

    for column in out.columns:
        name = str(column)
        try:
            numeric = pd.to_numeric(out[column], errors="coerce")
            rule = _domain_invalid_mask(name, numeric)
            if rule is None:
                continue
            mask, issue_type = rule
            count = int(mask.sum())
            if count == 0:
                continue

            before_series = out[column].copy()
            out.loc[mask, column] = pd.NA
            out[column] = pd.to_numeric(out[column], errors="coerce")
            after_series = out[column]
            samples = _sample_pairs(before_series, after_series, mask)
            invalid_indices[name] = out.index[mask]

            unique_before = [
                _jsonable(v) for v in before_series.loc[mask].dropna().unique()[:MAX_TRANSFORM_SAMPLES]
            ]

            _log(
                log,
                "invalid_value_handling",
                column=name,
                value_category="invalid_domain",
                issue_type=issue_type,
                before_invalid=count,
                after_invalid=0,
                action="convert_to_missing",
                before_values=unique_before,
                after_value=None,
                transformations=samples,
                message=(
                    f"Domain-invalid values in '{name}' ({issue_type}): "
                    f"{count} value(s) converted to missing so configured "
                    f"imputation can fill them. "
                    f"Examples before→after: "
                    + ", ".join(
                        f"{s['before']}→missing" for s in samples[:5]
                    )
                ),
            )
        except Exception as exc:
            _log(
                log,
                "invalid_value_handling_skipped",
                column=name,
                value_category="invalid_domain",
                error=str(exc),
                message="Invalid-value handling skipped for this column.",
            )
    return out, invalid_indices


def remove_duplicate_rows(
    df: pd.DataFrame,
    log: list[CleaningLogEntry],
) -> pd.DataFrame:
    before = _duplicate_row_count(df)
    if before == 0:
        _log(
            log,
            "duplicate_row_removal",
            duplicate_rows_found=0,
            rows_removed=0,
            message="No duplicate rows found.",
        )
        return df.copy()
    out = df.drop_duplicates().reset_index(drop=True)
    removed = int(df.shape[0] - out.shape[0])
    _log(
        log,
        "duplicate_row_removal",
        duplicate_rows_found=before,
        rows_removed=removed,
        message=f"Duplicate rows detected: {before}. Rows removed: {removed}.",
    )
    return out


def impute_missing_values(
    df: pd.DataFrame,
    log: list[CleaningLogEntry],
    *,
    high_missing_threshold: float,
    drop_high_missing: bool,
    invalid_indices: dict[str, pd.Index] | None = None,
    categorical_params: CategoricalImputationParams | None = None,
) -> pd.DataFrame:
    """Impute missing values; review / optionally drop high-missing columns.

    ``invalid_indices`` maps columns to row indices that became missing because
    of domain-invalid values. Those fills are logged distinctly from ordinary
    missingness while still using the same imputation strategy.
    """
    out = df.copy()
    n_rows = len(out)
    if n_rows == 0:
        return out

    invalid_indices = invalid_indices or {}
    drop_cols: list[str] = []
    cat_params = categorical_params or CategoricalImputationParams()

    for column in list(out.columns):
        series = out[column]
        missing_before = int(series.isna().sum())
        if missing_before == 0:
            continue

        missing_ratio = missing_before / n_rows
        col_name = str(column)
        from_invalid = invalid_indices.get(col_name)
        from_invalid_count = (
            int(series.loc[from_invalid].isna().sum()) if from_invalid is not None else 0
        )

        if missing_ratio > high_missing_threshold:
            if drop_high_missing:
                drop_cols.append(col_name)
                _log(
                    log,
                    "high_missing_column_dropped",
                    column=col_name,
                    missing_ratio=round(missing_ratio, 4),
                    threshold=high_missing_threshold,
                    before_missing=missing_before,
                    after_missing=0,
                    message=(
                        f"Dropped '{col_name}' due to high missingness "
                        f"({missing_ratio:.0%} > {high_missing_threshold:.0%})."
                    ),
                )
            else:
                _log(
                    log,
                    "high_missing_column_review",
                    column=col_name,
                    missing_ratio=round(missing_ratio, 4),
                    threshold=high_missing_threshold,
                    before_missing=missing_before,
                    after_missing=missing_before,
                    message=(
                        f"Marked '{col_name}' for review — missingness "
                        f"{missing_ratio:.0%} exceeds {high_missing_threshold:.0%}."
                    ),
                )
            continue

        if series.dropna().empty:
            _log(
                log,
                "all_null_column_review",
                column=col_name,
                before_missing=missing_before,
                after_missing=missing_before,
                message=f"Column '{col_name}' is entirely missing — left unchanged.",
            )
            continue

        missing_mask = series.isna()
        before_series = series.copy()

        try:
            if pd.api.types.is_numeric_dtype(series):
                integer_like = _is_integer_like(series)
                method = "median" if _is_skewed(series) else "mean"
                fill_value = (
                    float(series.median()) if method == "median" else float(series.mean())
                )
                out[column] = series.fillna(fill_value)
                if integer_like:
                    out[column], fill_value = _restore_integer_dtype(
                        out[column], fill_value
                    )
                final_dtype = str(out[column].dtype)
            elif pd.api.types.is_datetime64_any_dtype(series):
                mode = series.mode(dropna=True)
                if mode.empty:
                    _log(
                        log,
                        "missing_value_imputation_skipped",
                        column=col_name,
                        message="No mode available for datetime column.",
                    )
                    continue
                method = "mode"
                fill_value = mode.iloc[0]
                out[column] = series.fillna(fill_value)
                final_dtype = str(out[column].dtype)
            elif _is_categorical_series(series):
                out[column] = impute_categorical_column(
                    out,
                    col_name,
                    log,
                    cat_params,
                    missing_before=missing_before,
                    missing_ratio=missing_ratio,
                    from_invalid_count=from_invalid_count,
                )
                continue
            else:
                # Fallback for unexpected dtypes
                continue

            after_series = out[column]
            samples = _sample_pairs(before_series, after_series, missing_mask)
            after_missing = int(out[column].isna().sum())

            log_details: dict[str, Any] = {
                "method": method,
                "fill_value": _jsonable(fill_value),
                "before_missing": missing_before,
                "after_missing": after_missing,
                "from_invalid_domain_count": from_invalid_count,
                "before_dtype": str(series.dtype),
                "final_dtype": final_dtype,
                "integer_like": bool(
                    pd.api.types.is_integer_dtype(out[column])
                    if pd.api.types.is_numeric_dtype(out[column])
                    else False
                ),
                "before_values": [None] * min(len(samples), MAX_TRANSFORM_SAMPLES),
                "after_value": _jsonable(fill_value),
                "transformations": samples,
            }
            message = (
                f"Imputed {missing_before} missing value(s) in '{col_name}' "
                f"using {method} → {fill_value!r} "
                f"(dtype {series.dtype} → {final_dtype})"
                + (
                    f"; includes {from_invalid_count} former domain-invalid "
                    f"value(s) converted to missing."
                    if from_invalid_count
                    else "."
                )
            )

            _log(
                log,
                "missing_value_imputation",
                column=col_name,
                message=message,
                **log_details,
            )
        except Exception as exc:
            _log(
                log,
                "missing_value_imputation_skipped",
                column=col_name,
                error=str(exc),
                message="Missing-value imputation skipped for this column.",
            )

    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")
    return out


def handle_outliers(
    df: pd.DataFrame,
    log: list[CleaningLogEntry],
    strategy: OutlierStrategy,
) -> pd.DataFrame:
    """Detect *statistical* outliers via IQR (distinct from domain-invalid values).

    Default strategy FLAG reports them without changing values.
    """
    out = df.copy()
    if out.empty:
        return out

    rows_to_drop: set[Any] = set()

    for column in out.columns:
        series = out[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        bounds = _iqr_bounds(series)
        if bounds is None:
            continue
        lower, upper = bounds
        mask = _outlier_mask(series).fillna(False)
        count = int(mask.sum())
        if count == 0:
            continue

        pct = round((count / len(out)) * 100.0, 2)
        col_name = str(column)
        before_series = series.copy()

        if strategy == OutlierStrategy.FLAG:
            samples = _sample_pairs(before_series, before_series, mask)
            for sample in samples:
                sample["after"] = sample["before"]  # unchanged under FLAG
            unique_before = [
                _jsonable(v)
                for v in before_series.loc[mask].dropna().unique()[:MAX_TRANSFORM_SAMPLES]
            ]
            _log(
                log,
                "statistical_outlier_flagged",
                column=col_name,
                value_category="statistical_outlier",
                outlier_count=count,
                outlier_percentage=pct,
                lower_bound=round(lower, 4),
                upper_bound=round(upper, 4),
                strategy="flag",
                before_values=unique_before,
                after_values=unique_before,
                transformations=samples,
                message=(
                    f"Statistical outlier(s) in '{col_name}' (IQR): {count} "
                    f"({pct}%) flagged and retained. Not treated as domain-invalid. "
                    f"Examples: {', '.join(str(v) for v in unique_before[:5])}."
                ),
            )
        elif strategy == OutlierStrategy.CLIP:
            numeric = pd.to_numeric(out[column], errors="coerce")
            out[column] = numeric.clip(lower=lower, upper=upper)
            samples = _sample_pairs(before_series, out[column], mask)
            _log(
                log,
                "statistical_outlier_clipping",
                column=col_name,
                value_category="statistical_outlier",
                outlier_count=count,
                outlier_percentage=pct,
                lower_bound=round(lower, 4),
                upper_bound=round(upper, 4),
                strategy="clip",
                transformations=samples,
                message=(
                    f"Clipped {count} statistical outlier(s) in '{col_name}' "
                    f"to IQR bounds [{lower:.4g}, {upper:.4g}]."
                ),
            )
        elif strategy == OutlierStrategy.REMOVE:
            samples = _sample_pairs(before_series, before_series, mask)
            rows_to_drop.update(out.index[mask].tolist())
            _log(
                log,
                "statistical_outlier_removal_planned",
                column=col_name,
                value_category="statistical_outlier",
                outlier_count=count,
                outlier_percentage=pct,
                lower_bound=round(lower, 4),
                upper_bound=round(upper, 4),
                strategy="remove",
                before_values=[
                    _jsonable(v)
                    for v in before_series.loc[mask].dropna().unique()[:MAX_TRANSFORM_SAMPLES]
                ],
                after_value="row_removed",
                transformations=samples,
                message=(
                    f"Marked {count} statistical outlier row(s) from '{col_name}' "
                    f"for removal."
                ),
            )

    if strategy == OutlierStrategy.REMOVE and rows_to_drop:
        before_rows = len(out)
        out = out.drop(index=list(rows_to_drop)).reset_index(drop=True)
        _log(
            log,
            "statistical_outlier_row_removal",
            value_category="statistical_outlier",
            rows_removed=before_rows - len(out),
            strategy="remove",
            before_rows=before_rows,
            after_rows=len(out),
            message=f"Removed {before_rows - len(out)} row(s) containing statistical outliers.",
        )

    return out


def remove_constant_columns(
    df: pd.DataFrame,
    log: list[CleaningLogEntry],
) -> pd.DataFrame:
    constants = _constant_columns(df)
    if not constants:
        _log(
            log,
            "constant_column_removal",
            columns_removed=[],
            message="No constant columns to remove.",
        )
        return df.copy()
    out = df.drop(columns=constants, errors="ignore")
    _log(
        log,
        "constant_column_removal",
        columns_removed=constants,
        message=f"Removed constant column(s): {', '.join(constants)}.",
    )
    return out


def retain_constant_columns_note(
    df: pd.DataFrame,
    log: list[CleaningLogEntry],
) -> None:
    constants = _constant_columns(df)
    if constants:
        _log(
            log,
            "constant_columns_retained",
            columns=constants,
            message=(
                f"Constant column(s) retained: {', '.join(constants)}. "
                "Enable 'Remove constant columns' to drop them."
            ),
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_cleaning_pipeline(
    df: pd.DataFrame,
    config: CleaningConfig,
) -> tuple[pd.DataFrame, list[CleaningLogEntry], list[str]]:
    """Apply configured cleaning steps to a copy of ``df``.

    Returns cleaned frame, structured log, and list of modified columns.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise DataCleanerError("Dataset could not be loaded into a DataFrame.")

    original = df.copy()
    working = _safe_column_names(original)
    log: list[CleaningLogEntry] = []
    modified: set[str] = set()

    if working.empty and len(working.columns) == 0:
        raise DataCleanerError("The dataset is empty and cannot be cleaned.")

    if config.handle_empty_strings:
        before_missing = _total_missing(working)
        working = normalize_empty_strings(working, log)
        if _total_missing(working) != before_missing:
            modified.update(
                e.column for e in log if e.operation == "empty_string_normalization" and e.column
            )

    if config.convert_safe_dtypes:
        working = convert_safe_dtypes(working, log)
        modified.update(
            e.column for e in log if e.operation == "dtype_conversion" and e.column
        )

    if config.handle_invalid_values:
        working, invalid_indices = handle_invalid_numeric_values(working, log)
        modified.update(
            e.column
            for e in log
            if e.operation == "invalid_value_handling" and e.column
        )
    else:
        invalid_indices = {}

    working = process_duplicate_columns(
        working,
        log,
        remove=config.remove_duplicate_columns,
    )
    if config.remove_duplicate_columns:
        modified.update(
            e.column
            for e in log
            if e.operation == "duplicate_column_removal" and e.column
        )

    # Impute before deduplication so domain-invalid→missing cells get filled
    # with the configured strategy and index tracking stays aligned.
    cat_params = CategoricalImputationParams.from_cleaning_config(config)
    if config.handle_missing_values:
        working = impute_missing_values(
            working,
            log,
            high_missing_threshold=config.high_missingness_threshold,
            drop_high_missing=config.drop_high_missing_columns,
            invalid_indices=invalid_indices,
            categorical_params=cat_params,
        )
        modified.update(
            e.column
            for e in log
            if e.operation
            in {
                "missing_value_imputation",
                "high_missing_column_dropped",
            }
            and e.column
        )

    if config.remove_duplicate_rows:
        before_rows = len(working)
        working = remove_duplicate_rows(working, log)
        if len(working) != before_rows:
            modified.add("__rows__")

    if config.handle_outliers:
        before_rows = len(working)
        working = handle_outliers(working, log, config.outlier_strategy)
        if config.outlier_strategy == OutlierStrategy.CLIP:
            modified.update(
                e.column
                for e in log
                if e.operation == "statistical_outlier_clipping" and e.column
            )
        if config.outlier_strategy == OutlierStrategy.REMOVE and len(working) != before_rows:
            modified.add("__rows__")

    if config.remove_constant_columns:
        before_cols = set(working.columns.astype(str))
        working = remove_constant_columns(working, log)
        dropped = before_cols - set(working.columns.astype(str))
        modified.update(dropped)
    else:
        retain_constant_columns_note(working, log)

    # Identifier columns are intentionally never auto-removed.
    _log(
        log,
        "identifier_columns_policy",
        message=(
            "Identifier-like columns (e.g. Customer_ID) are retained. "
            "Feature selection is deferred to a later phase."
        ),
    )

    columns_modified = sorted(c for c in modified if c != "__rows__")
    return working, log, columns_modified


def build_cleaning_summary(
    before: DatasetSnapshot,
    after: DatasetSnapshot,
    log: list[CleaningLogEntry],
    config: CleaningConfig,
) -> CleaningSummary:
    """Translate pipeline log + snapshots into a user-facing summary."""
    issues_found: list[str] = []
    to_fix: list[IssueTransition] = []
    to_flag: list[IssueTransition] = []

    duplicate_cols = [
        e for e in log if e.operation == "duplicate_column_removal"
    ]
    duplicate_detected = [
        e for e in log if e.operation == "duplicate_column_detection"
    ]
    dup_count = 0
    if duplicate_detected:
        dup_count = int(duplicate_detected[0].details.get("detected_count", 0))
    elif duplicate_cols:
        dup_count = len(duplicate_cols)
    if dup_count:
        issues_found.append(f"Duplicate columns: {dup_count}")
        if config.remove_duplicate_columns:
            to_fix.append(
                IssueTransition(
                    issue="Duplicate columns",
                    before=dup_count,
                    after=0,
                    action="removed redundant columns",
                )
            )
        else:
            to_flag.append(
                IssueTransition(
                    issue="Duplicate columns",
                    before=dup_count,
                    after=dup_count,
                    action="retained (option off)",
                )
            )

    if before.duplicate_rows:
        issues_found.append(f"Duplicate rows: {before.duplicate_rows}")
        if config.remove_duplicate_rows:
            to_fix.append(
                IssueTransition(
                    issue="Duplicate rows",
                    before=before.duplicate_rows,
                    after=after.duplicate_rows,
                    action="removed",
                )
            )
        else:
            to_flag.append(
                IssueTransition(
                    issue="Duplicate rows",
                    before=before.duplicate_rows,
                    after=before.duplicate_rows,
                    action="retained (option off)",
                )
            )

    if before.missing_values:
        issues_found.append(f"Missing values: {before.missing_values}")
        if config.handle_missing_values:
            to_fix.append(
                IssueTransition(
                    issue="Missing values",
                    before=before.missing_values,
                    after=after.missing_values,
                    action="imputed / reviewed",
                )
            )
        else:
            to_flag.append(
                IssueTransition(
                    issue="Missing values",
                    before=before.missing_values,
                    after=before.missing_values,
                    action="retained (option off)",
                )
            )

    invalid_entries = [
        e for e in log if e.operation == "invalid_value_handling"
    ]
    if invalid_entries:
        total_invalid = sum(int(e.details.get("before_invalid", 0)) for e in invalid_entries)
        issues_found.append(f"Domain-invalid values: {total_invalid}")
        to_fix.append(
            IssueTransition(
                issue="Domain-invalid values (→ missing → impute)",
                before=total_invalid,
                after=0,
                action="converted to missing, then imputed if enabled",
            )
        )

    dtype_entries = [e for e in log if e.operation == "dtype_conversion"]
    if dtype_entries:
        issues_found.append(f"Unsafe / text dtypes converted: {len(dtype_entries)}")
        to_fix.append(
            IssueTransition(
                issue="Data type conversions",
                before=len(dtype_entries),
                after=0,
                action="converted safely",
            )
        )

    outlier_flagged = [
        e
        for e in log
        if e.operation == "statistical_outlier_flagged"
    ]
    outlier_fixed = [
        e
        for e in log
        if e.operation
        in {
            "statistical_outlier_clipping",
            "statistical_outlier_removal_planned",
            "statistical_outlier_row_removal",
        }
    ]
    outlier_count = sum(
        int(e.details.get("outlier_count", 0))
        for e in log
        if e.operation
        in {
            "statistical_outlier_flagged",
            "statistical_outlier_clipping",
            "statistical_outlier_removal_planned",
        }
    )
    if outlier_count:
        issues_found.append(f"Statistical outliers (IQR): {outlier_count}")
        if config.outlier_strategy == OutlierStrategy.FLAG or outlier_flagged:
            to_flag.append(
                IssueTransition(
                    issue="Statistical outliers (IQR)",
                    before=outlier_count,
                    after=outlier_count,
                    action="flagged (retained)",
                )
            )
        elif outlier_fixed:
            to_fix.append(
                IssueTransition(
                    issue="Statistical outliers (IQR)",
                    before=outlier_count,
                    after=0 if config.outlier_strategy == OutlierStrategy.REMOVE else "clipped",
                    action=config.outlier_strategy.value,
                )
            )

    review_cols = [
        e.column
        for e in log
        if e.operation
        in {
            "high_missing_column_review",
            "all_null_column_review",
            "categorical_imputation_review",
        }
        and e.column
    ]
    for col in review_cols:
        to_flag.append(
            IssueTransition(
                issue=f"High/all missing column '{col}'",
                before="review",
                after="review",
                action="flagged for review",
            )
        )

    retained_constants = next(
        (e for e in log if e.operation == "constant_columns_retained"),
        None,
    )
    removed_constants = next(
        (e for e in log if e.operation == "constant_column_removal"),
        None,
    )
    if retained_constants:
        cols = retained_constants.details.get("columns", [])
        issues_found.append(f"Constant columns: {len(cols)}")
        to_flag.append(
            IssueTransition(
                issue="Constant columns",
                before=len(cols),
                after=len(cols),
                action="retained",
            )
        )
    elif removed_constants and removed_constants.details.get("columns_removed"):
        cols = removed_constants.details["columns_removed"]
        issues_found.append(f"Constant columns: {len(cols)}")
        to_fix.append(
            IssueTransition(
                issue="Constant columns",
                before=len(cols),
                after=0,
                action="removed",
            )
        )

    return CleaningSummary(
        issues_found=issues_found or ["No major issues detected."],
        issues_to_fix=to_fix,
        issues_to_flag=to_flag,
    )


def _load_dataset(db: Session, dataset_id: int) -> tuple[Dataset, pd.DataFrame]:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise DataCleanerError(f"Dataset with id {dataset_id} was not found.")
    path = Path(dataset.dataset_path)
    if not path.is_file():
        raise DataCleanerError("Original dataset file is missing on the server.")
    try:
        df = read_dataset_csv(path)
    except DatasetServiceError as exc:
        raise DataCleanerError(str(exc)) from exc
    return dataset, df


def _cleaned_output_path(original_filename: str) -> Path:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(original_filename).stem
    suffix = Path(original_filename).suffix or ".csv"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem) or "dataset"
    return settings.processed_dir / f"cleaned_{safe_stem}{suffix}"


def preview_cleaning(
    db: Session,
    dataset_id: int,
    config: CleaningConfig,
) -> CleaningPreviewResponse:
    """Dry-run cleaning without writing any files."""
    dataset, df = _load_dataset(db, dataset_id)
    original_path = Path(dataset.dataset_path)
    original_bytes_before = original_path.read_bytes()

    before = _snapshot(df)
    cleaned, log, modified = run_cleaning_pipeline(df, config)
    after = _snapshot(cleaned, columns_modified=modified)
    summary = build_cleaning_summary(before, after, log, config)

    if original_path.read_bytes() != original_bytes_before:
        raise DataCleanerError("Safety check failed: original file was modified.")

    return CleaningPreviewResponse(
        dataset_id=dataset_id,
        config=config,
        summary=summary,
        before=before,
        planned_log=log,
    )


def apply_cleaning(
    db: Session,
    dataset_id: int,
    config: CleaningConfig,
) -> CleaningApplyResponse:
    """Run cleaning, save a cleaned copy under ``processed/``, return results."""
    dataset, df = _load_dataset(db, dataset_id)
    original_path = Path(dataset.dataset_path)
    original_bytes_before = original_path.read_bytes()

    before = _snapshot(df)
    cleaned, log, modified = run_cleaning_pipeline(df, config)
    after = _snapshot(cleaned, columns_modified=modified)
    summary = build_cleaning_summary(before, after, log, config)

    destination = _cleaned_output_path(dataset.filename)
    try:
        cleaned.to_csv(destination, index=False)
    except Exception as exc:
        raise DataCleanerError(f"Failed to save cleaned dataset: {exc}") from exc

    if original_path.read_bytes() != original_bytes_before:
        # Extremely defensive — should never happen
        raise DataCleanerError("Safety check failed: original file was modified.")

    _log(
        log,
        "cleaned_dataset_saved",
        cleaned_path=str(destination),
        message=f"Cleaned dataset saved to {destination.name}.",
    )

    return CleaningApplyResponse(
        dataset_id=dataset_id,
        config=config,
        summary=summary,
        before=before,
        after=after,
        cleaning_log=log,
        cleaned_filename=destination.name,
        cleaned_path=str(destination),
        preview=_build_preview_rows(cleaned),
        download_url=f"/api/v1/datasets/{dataset_id}/clean/download",
    )


def get_cleaned_file_path(db: Session, dataset_id: int) -> Path:
    """Resolve the expected cleaned file path for a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise DataCleanerError(f"Dataset with id {dataset_id} was not found.")
    path = _cleaned_output_path(dataset.filename)
    if not path.is_file():
        raise DataCleanerError(
            "No cleaned dataset found. Apply cleaning first."
        )
    return path
