"""Phase 5.1 — Feature type detection and engineering opportunity recommendations.

Read-only analysis. Does NOT create, delete, encode, or transform columns.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.schemas.feature_engineering import (
    FeatureColumnAnalysis,
    FeatureEngineeringOpportunityReport,
    FeatureOpportunity,
    FeatureRelationshipOpportunity,
)
from app.services.data_cleaner import get_cleaned_file_path, DataCleanerError
from app.services.dataset_profiler import (
    _is_boolean_like,
    _is_datetime_like,
    _name_matches_identifier,
    _name_matches_target,
)
from app.services.dataset_service import DatasetServiceError, read_dataset_csv

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------

LOW_CARDINALITY_MAX = 10
MEDIUM_CARDINALITY_MAX = 50
HIGH_UNIQUENESS_RATIO = 0.95
NEAR_CONSTANT_DOMINANT_SHARE = 0.98
SKEW_LOG_THRESHOLD = 1.0
AGE_BINNING_MIN = 0
AGE_BINNING_MAX = 120
SHORT_TEXT_MEAN_LEN = 40
LONG_TEXT_MEAN_LEN = 80
DATETIME_PARSE_RATIO = 0.8
MIN_ROWS_FOR_STATS = 3


_HEIGHT_NAME = re.compile(r"(^|_)(height|ht)($|_)", re.IGNORECASE)
_WEIGHT_NAME = re.compile(r"(^|_)(weight|wt|mass)($|_)", re.IGNORECASE)
_SALARY_NAME = re.compile(
    r"salary|wage|income|compensation|charges|revenue", re.IGNORECASE
)
_EXPERIENCE_NAME = re.compile(
    r"experience|tenure|years.?exp|yrs.?exp", re.IGNORECASE
)
_AGE_NAME = re.compile(r"(^|_)age($|_)", re.IGNORECASE)
_START_DATE_NAME = re.compile(
    r"(start|begin|join|hire|registration|signup|created).*(date|time|at)|"
    r"^(join_date|start_date|hire_date|created_at)$",
    re.IGNORECASE,
)
_END_DATE_NAME = re.compile(
    r"(end|exit|leave|terminat|churn_date|closed).*(date|time|at)|"
    r"^(end_date|exit_date|closed_at)$",
    re.IGNORECASE,
)
_TEXT_NAME = re.compile(
    r"review|comment|description|desc|notes?|feedback|address|"
    r"bio|summary|message|text|content",
    re.IGNORECASE,
)
_NAME_COLUMN = re.compile(r"(^|_)(name|full.?name|first.?name|last.?name)($|_)", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class FeatureOpportunityError(Exception):
    """Raised when feature-opportunity analysis cannot complete."""


# ---------------------------------------------------------------------------
# Core detectors
# ---------------------------------------------------------------------------


def detect_identifier(series: pd.Series, column_name: str) -> dict[str, Any] | None:
    """Return identifier evidence or None."""
    n_rows = len(series)
    non_null = series.dropna()
    nunique = int(non_null.nunique())
    reasons: list[str] = []

    name_hit = _name_matches_identifier(column_name)
    if name_hit:
        reasons.append("Column name matches identifier patterns.")

    # Person-name columns are unique but are text features, not IDs.
    if _NAME_COLUMN.search(column_name) and not name_hit:
        return None

    uniqueness = (nunique / n_rows) if n_rows else 0.0
    if n_rows > 1 and uniqueness >= HIGH_UNIQUENESS_RATIO and nunique > 1:
        reasons.append(
            f"Very high uniqueness ({nunique:,} of {n_rows:,} rows are distinct)."
        )

    # Continuous floats are almost always unique — not an ID signal alone.
    is_numeric = pd.api.types.is_numeric_dtype(non_null)
    integer_like = False
    if is_numeric and not non_null.empty:
        vals = non_null.to_numpy(dtype=float)
        integer_like = bool(np.allclose(vals, np.round(vals)))

    # Sequential numeric IDs (integers only)
    sequential = False
    if is_numeric and integer_like and nunique >= max(10, int(0.9 * n_rows)):
        sorted_vals = np.sort(non_null.to_numpy(dtype=float))
        if len(sorted_vals) >= 3:
            diffs = np.diff(sorted_vals)
            if np.all(diffs >= 0) and float(np.median(diffs)) in (0.0, 1.0) and uniqueness >= 0.9:
                sequential = True
                reasons.append("Appears sequential / nearly unique numeric identifier.")

    # Email-like uniqueness
    email_like = False
    if non_null.dtype == object or pd.api.types.is_string_dtype(non_null):
        sample = non_null.astype(str).head(min(50, len(non_null)))
        email_hits = sum(1 for v in sample if _EMAIL_PATTERN.match(v.strip()))
        if len(sample) and email_hits / len(sample) >= 0.8 and uniqueness >= 0.9:
            email_like = True
            reasons.append("Values look like unique email addresses.")

    # Name match + meaningful uniqueness
    if name_hit and uniqueness >= 0.5:
        return {
            "is_identifier": True,
            "uniqueness_ratio": round(uniqueness, 4),
            "unique_count": nunique,
            "reasons": reasons,
        }

    # String / object high uniqueness without measure semantics
    if (
        not is_numeric
        and uniqueness >= HIGH_UNIQUENESS_RATIO
        and nunique > max(5, int(0.5 * n_rows))
    ):
        return {
            "is_identifier": True,
            "uniqueness_ratio": round(uniqueness, 4),
            "unique_count": nunique,
            "reasons": reasons or ["Extremely high uniqueness suggests an identifier."],
        }

    # Integer sequential IDs without requiring name (e.g. 1001, 1002, ...)
    if sequential and uniqueness >= HIGH_UNIQUENESS_RATIO:
        return {
            "is_identifier": True,
            "uniqueness_ratio": round(uniqueness, 4),
            "unique_count": nunique,
            "reasons": reasons,
        }

    if email_like:
        return {
            "is_identifier": True,
            "uniqueness_ratio": round(uniqueness, 4),
            "unique_count": nunique,
            "reasons": reasons,
        }

    return None


def detect_datetime_feature(series: pd.Series, column_name: str) -> dict[str, Any] | None:
    """Detect datetime dtype or parseable date strings."""
    if pd.api.types.is_datetime64_any_dtype(series):
        non_null = series.dropna()
        return {
            "is_datetime": True,
            "source": "dtype",
            "unique_count": int(non_null.nunique()),
            "parse_ratio": 1.0,
        }

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return None

    if _is_datetime_like(series):
        non_null = series.dropna()
        sample = non_null.head(min(200, len(non_null)))
        parsed = pd.to_datetime(sample, errors="coerce")
        parse_ratio = float(parsed.notna().mean()) if len(sample) else 0.0
        return {
            "is_datetime": True,
            "source": "parsed_string",
            "unique_count": int(non_null.nunique()),
            "parse_ratio": round(parse_ratio, 4),
        }
    return None


def detect_boolean_feature(series: pd.Series) -> bool:
    return _is_boolean_like(series)


def detect_text_feature(series: pd.Series, column_name: str) -> dict[str, Any] | None:
    """Distinguish short/long text, names, free-form content."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return None
    if pd.api.types.is_datetime64_any_dtype(series):
        return None

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None

    nunique = int(non_null.nunique())
    n_rows = len(series)
    uniqueness = nunique / n_rows if n_rows else 0.0
    lengths = non_null.str.len()
    mean_len = float(lengths.mean())
    name_suggests_text = bool(_TEXT_NAME.search(column_name))
    name_suggests_name = bool(_NAME_COLUMN.search(column_name))

    # Low-cardinality object → categorical, not text
    if nunique <= MEDIUM_CARDINALITY_MAX and uniqueness < 0.5 and not name_suggests_text:
        if mean_len < SHORT_TEXT_MEAN_LEN:
            return None

    if name_suggests_name and mean_len < LONG_TEXT_MEAN_LEN:
        return {
            "is_text": True,
            "text_kind": "names",
            "mean_length": round(mean_len, 1),
            "unique_count": nunique,
        }

    if name_suggests_text or mean_len >= LONG_TEXT_MEAN_LEN or (
        uniqueness >= 0.7 and mean_len >= SHORT_TEXT_MEAN_LEN
    ):
        kind = "long/free-form text" if mean_len >= LONG_TEXT_MEAN_LEN else "short text"
        if "review" in column_name.lower():
            kind = "reviews"
        elif "address" in column_name.lower():
            kind = "addresses"
        elif "comment" in column_name.lower() or "note" in column_name.lower():
            kind = "comments"
        elif "desc" in column_name.lower():
            kind = "descriptions"
        return {
            "is_text": True,
            "text_kind": kind,
            "mean_length": round(mean_len, 1),
            "unique_count": nunique,
        }
    return None


def _cardinality_label(n_unique: int) -> str:
    if n_unique <= LOW_CARDINALITY_MAX:
        return "Low cardinality"
    if n_unique <= MEDIUM_CARDINALITY_MAX:
        return "Medium cardinality"
    return "High cardinality"


def analyze_numeric_feature(series: pd.Series, column_name: str) -> dict[str, Any]:
    """Numeric profile + opportunity candidates (recommendations only)."""
    non_null = pd.to_numeric(series, errors="coerce").dropna()
    n_rows = len(series)
    missing_pct = round((1 - len(non_null) / n_rows) * 100.0, 2) if n_rows else 0.0
    nunique = int(non_null.nunique())

    if non_null.empty:
        return {
            "dtype": str(series.dtype),
            "unique_values": 0,
            "missing_percentage": missing_pct,
            "integer_like": False,
            "opportunities": [],
        }

    skew = float(non_null.skew()) if len(non_null) >= MIN_ROWS_FOR_STATS else 0.0
    integer_like = bool(
        np.allclose(non_null.to_numpy(dtype=float), np.round(non_null.to_numpy(dtype=float)))
    )

    profile: dict[str, Any] = {
        "dtype": str(series.dtype),
        "unique_values": nunique,
        "missing_percentage": missing_pct,
        "min": float(non_null.min()),
        "max": float(non_null.max()),
        "mean": float(non_null.mean()),
        "median": float(non_null.median()),
        "std": float(non_null.std(ddof=0)) if len(non_null) else 0.0,
        "skewness": round(skew, 4),
        "integer_like": integer_like,
        "opportunities": [],
    }

    ops: list[FeatureOpportunity] = []

    if _AGE_NAME.search(column_name) and AGE_BINNING_MIN <= profile["min"] and profile["max"] <= AGE_BINNING_MAX:
        ops.append(
            FeatureOpportunity(
                opportunity="Possible binning",
                priority="MEDIUM",
                reason=(
                    f"{column_name} looks like an age-like numeric feature "
                    f"(range {profile['min']:.0f}–{profile['max']:.0f}); "
                    "age bins can capture non-linear effects."
                ),
            )
        )

    if abs(skew) >= SKEW_LOG_THRESHOLD and profile["min"] >= 0:
        ops.append(
            FeatureOpportunity(
                opportunity="Possible log / skewness transformation",
                priority="MEDIUM" if abs(skew) < 2 else "HIGH",
                reason=(
                    f"Distribution is strongly skewed (skewness={skew:.2f}); "
                    "a log or power transform may stabilize variance."
                ),
            )
        )

    profile["opportunities"] = ops
    return profile


def analyze_categorical_feature(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    nunique = int(non_null.nunique())
    counts = non_null.astype(str).value_counts(dropna=True)
    dominant = counts.index[0] if len(counts) else None
    dominant_pct = round(float(counts.iloc[0] / len(non_null) * 100.0), 2) if len(counts) else 0.0
    cardinality = _cardinality_label(nunique)

    ops: list[FeatureOpportunity] = []
    if nunique <= LOW_CARDINALITY_MAX:
        ops.append(
            FeatureOpportunity(
                opportunity="Encoding",
                priority="HIGH",
                reason=(
                    f"Low-cardinality categorical feature ({nunique} categories); "
                    "one-hot or ordinal encoding is typically appropriate."
                ),
            )
        )
    elif nunique <= MEDIUM_CARDINALITY_MAX:
        ops.append(
            FeatureOpportunity(
                opportunity="Encoding / rare-category grouping",
                priority="MEDIUM",
                reason=(
                    f"Medium cardinality ({nunique} categories); "
                    "consider grouping rare levels before encoding."
                ),
            )
        )
    else:
        ops.append(
            FeatureOpportunity(
                opportunity="Category frequency encoding",
                priority="MEDIUM",
                reason=(
                    f"High cardinality ({nunique} categories); "
                    "frequency or hashing encodings may be more practical than one-hot."
                ),
            )
        )

    if dominant_pct >= 80 and nunique > 1:
        ops.append(
            FeatureOpportunity(
                opportunity="Rare-category grouping",
                priority="MEDIUM",
                reason=(
                    f"Dominant category '{dominant}' is {dominant_pct:.1f}% of values; "
                    "tail categories may benefit from grouping."
                ),
            )
        )

    return {
        "unique_categories": nunique,
        "cardinality": cardinality,
        "dominant_category": dominant,
        "dominant_percentage": dominant_pct,
        "opportunities": ops,
    }


def detect_feature_type(series: pd.Series, column_name: str) -> tuple[str, dict[str, Any]]:
    """Classify a column into a semantic feature type."""
    n_rows = len(series)
    non_null = series.dropna()
    nunique = int(non_null.nunique())

    # Constant / near-constant first (except pure boolean binaries)
    if nunique <= 1 and not non_null.empty:
        return "constant / near-constant", {
            "unique_values": nunique,
            "dominant_share": 1.0,
        }

    if not non_null.empty and nunique >= 1:
        top_share = float(non_null.astype(str).value_counts(normalize=True).iloc[0])
        if top_share >= NEAR_CONSTANT_DOMINANT_SHARE and nunique > 1:
            if not detect_boolean_feature(series):
                return "constant / near-constant", {
                    "unique_values": nunique,
                    "dominant_share": round(top_share, 4),
                    "dominant_category": str(non_null.astype(str).value_counts().index[0]),
                }

    ident = detect_identifier(series, column_name)
    if ident:
        return "identifier", ident

    if detect_boolean_feature(series):
        return "boolean", {"unique_values": nunique}

    dt = detect_datetime_feature(series, column_name)
    if dt:
        return "datetime", dt

    text = detect_text_feature(series, column_name)
    if text:
        return "text", text

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "numerical", analyze_numeric_feature(series, column_name)

    # Numeric stored as object but mostly numeric
    if non_null.dtype == object or pd.api.types.is_string_dtype(non_null):
        coerced = pd.to_numeric(non_null, errors="coerce")
        if len(non_null) and coerced.notna().mean() >= 0.9:
            return "numerical", analyze_numeric_feature(series, column_name)

        if nunique <= MEDIUM_CARDINALITY_MAX or (n_rows and nunique / n_rows < 0.5):
            return "categorical", analyze_categorical_feature(series)

        # High uniqueness object without ID name → still often categorical/text
        if text is None and nunique > MEDIUM_CARDINALITY_MAX:
            return "categorical", analyze_categorical_feature(series)

    if pd.api.types.is_categorical_dtype(series):
        return "categorical", analyze_categorical_feature(series)

    return "unknown / other", {"unique_values": nunique, "dtype": str(series.dtype)}


def calculate_opportunity_priority(
    feature_type: str,
    opportunities: list[FeatureOpportunity],
) -> str:
    if not opportunities:
        if feature_type == "identifier":
            return "HIGH"
        return "LOW"
    order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    return max(opportunities, key=lambda o: order.get(o.priority, 0)).priority


def _primary_opportunity_summary(
    feature_type: str,
    details: dict[str, Any],
) -> tuple[str, str, str, list[FeatureOpportunity]]:
    """Return opportunity label, priority, reason, and opportunity list."""
    ops: list[FeatureOpportunity] = list(details.get("opportunities") or [])

    if feature_type == "identifier":
        reason = " ".join(details.get("reasons") or [])
        if not reason:
            reason = "Likely identifier; mathematical transformations are not meaningful."
        return (
            "None",
            "HIGH",
            reason + " Do not perform mathematical feature engineering.",
            [],
        )

    if feature_type == "constant / near-constant":
        share = details.get("dominant_share", 1.0)
        return (
            "Potentially low-information feature",
            "MEDIUM",
            (
                f"Near-constant / low variance (dominant share ≈ {share:.1%}). "
                "Do not auto-remove here; feature selection is deferred."
            ),
            [
                FeatureOpportunity(
                    opportunity="Potentially low-information feature",
                    priority="MEDIUM",
                    reason="Feature has little variation across rows.",
                )
            ],
        )

    if feature_type == "datetime":
        op = FeatureOpportunity(
            opportunity="Datetime decomposition",
            priority="HIGH",
            reason=(
                "Datetime feature engineering opportunity detected. "
                "Year, month, day, day-of-week, and related temporal components "
                "can be extracted later — not created in Step 5.1."
            ),
        )
        return op.opportunity, op.priority, op.reason, [op]

    if feature_type == "boolean":
        return (
            "None (already binary)",
            "LOW",
            "Column already represents binary information; unnecessary transforms are not recommended.",
            [],
        )

    if feature_type == "text":
        kind = details.get("text_kind", "text")
        op = FeatureOpportunity(
            opportunity="Text-derived features",
            priority="MEDIUM",
            reason=(
                f"Detected as {kind} (mean length ≈ {details.get('mean_length', '?')}). "
                "Character/word counts or later NLP features may help — not applied yet."
            ),
        )
        return op.opportunity, op.priority, op.reason, [op]

    if feature_type == "numerical":
        if ops:
            primary = ops[0]
            return primary.opportunity, primary.priority, primary.reason, ops
        return (
            "None",
            "LOW",
            "Numerical feature with no strong data-driven transform recommended at this stage.",
            [],
        )

    if feature_type == "categorical":
        if ops:
            primary = ops[0]
            return primary.opportunity, primary.priority, primary.reason, ops
        return "Encoding", "MEDIUM", "Categorical feature may benefit from encoding.", []

    return "None", "LOW", "No clear feature-engineering opportunity.", []


def _characteristics_summary(feature_type: str, details: dict[str, Any]) -> str:
    if feature_type == "numerical":
        parts = [
            f"dtype={details.get('dtype')}",
            f"{details.get('unique_values', 0)} unique",
        ]
        if "skewness" in details:
            parts.append(f"skew={details['skewness']}")
        if details.get("integer_like"):
            parts.append("integer-like")
        else:
            parts.append("continuous")
        return "; ".join(parts)

    if feature_type == "categorical":
        return (
            f"{details.get('unique_categories', 0)} categories "
            f"({details.get('cardinality')}); "
            f"dominant={details.get('dominant_category')!r} "
            f"({details.get('dominant_percentage')}%)"
        )

    if feature_type == "datetime":
        return f"{details.get('unique_count', 0):,} unique dates/timestamps"

    if feature_type == "identifier":
        ratio = details.get("uniqueness_ratio")
        pct = f"{ratio * 100:.1f}%" if ratio is not None else "high"
        return f"{details.get('unique_count', 0)} unique ({pct} uniqueness)"

    if feature_type == "text":
        return (
            f"{details.get('text_kind')}; mean length={details.get('mean_length')}; "
            f"{details.get('unique_count')} unique"
        )

    if feature_type == "boolean":
        return f"{details.get('unique_values', 2)} distinct binary values"

    if feature_type == "constant / near-constant":
        return (
            f"{details.get('unique_values', 1)} unique; "
            f"dominant share={details.get('dominant_share', 1.0)}"
        )

    return str(details.get("dtype", "—"))


def detect_feature_relationships(
    df: pd.DataFrame,
    type_map: dict[str, str],
) -> list[FeatureRelationshipOpportunity]:
    """Heuristic relationship opportunities — never invent all-pairs combos."""
    results: list[FeatureRelationshipOpportunity] = []
    cols = [str(c) for c in df.columns]

    height_cols = [c for c in cols if _HEIGHT_NAME.search(c) and type_map.get(c) == "numerical"]
    weight_cols = [c for c in cols if _WEIGHT_NAME.search(c) and type_map.get(c) == "numerical"]
    for h in height_cols:
        for w in weight_cols:
            results.append(
                FeatureRelationshipOpportunity(
                    columns=[h, w],
                    opportunity="Possible BMI / body-composition ratio",
                    priority="HIGH",
                    reason=(
                        f"Columns '{h}' and '{w}' suggest height and weight; "
                        "BMI-style features may be meaningful later."
                    ),
                )
            )

    salary_cols = [
        c for c in cols if _SALARY_NAME.search(c) and type_map.get(c) == "numerical"
    ]
    exp_cols = [
        c for c in cols if _EXPERIENCE_NAME.search(c) and type_map.get(c) in {"numerical", "categorical"}
    ]
    for s in salary_cols:
        for e in exp_cols:
            results.append(
                FeatureRelationshipOpportunity(
                    columns=[s, e],
                    opportunity="Possible salary-per-experience ratio",
                    priority="MEDIUM",
                    reason=(
                        f"'{s}' and '{e}' appear related; "
                        "a ratio feature may capture compensation relative to experience."
                    ),
                )
            )

    start_cols = [
        c for c in cols if _START_DATE_NAME.search(c) and type_map.get(c) == "datetime"
    ]
    end_cols = [
        c for c in cols if _END_DATE_NAME.search(c) and type_map.get(c) == "datetime"
    ]
    for s in start_cols:
        for e in end_cols:
            if s == e:
                continue
            results.append(
                FeatureRelationshipOpportunity(
                    columns=[s, e],
                    opportunity="Possible duration / tenure",
                    priority="HIGH",
                    reason=(
                        f"'{s}' and '{e}' look like start/end timestamps; "
                        "duration can be derived later."
                    ),
                )
            )

    # Single join/start date → tenure vs "today" (reference date)
    if start_cols and not end_cols:
        for s in start_cols:
            results.append(
                FeatureRelationshipOpportunity(
                    columns=[s],
                    opportunity="Possible tenure vs reference date",
                    priority="MEDIUM",
                    reason=(
                        f"'{s}' is a start/join datetime; "
                        "tenure relative to a reference date may be useful later."
                    ),
                )
            )

    return results


def generate_feature_engineering_opportunities(
    df: pd.DataFrame,
) -> tuple[list[FeatureColumnAnalysis], list[FeatureRelationshipOpportunity], list[str]]:
    """Analyze all columns; return analyses, relationships, potential targets."""
    analyses: list[FeatureColumnAnalysis] = []
    type_map: dict[str, str] = {}
    potential_targets: list[str] = []

    for column in df.columns:
        col_name = str(column)
        series = df[column]
        feature_type, details = detect_feature_type(series, col_name)
        type_map[col_name] = feature_type

        is_target = _name_matches_target(col_name)
        if is_target:
            potential_targets.append(col_name)

        opportunity, priority, reason, ops = _primary_opportunity_summary(
            feature_type, details
        )

        # Avoid recommending FE that would leak target-derived info
        leakage = False
        if is_target:
            leakage = True
            opportunity = "None (potential target - leakage-sensitive)"
            priority = "HIGH"
            reason = (
                f"'{col_name}' looks like a target/label column. "
                "Do not engineer features from the target itself."
            )
            ops = []

        # Serialize opportunities for details without nested pydantic issues
        detail_out = {
            k: v
            for k, v in details.items()
            if k != "opportunities"
        }
        if ops:
            detail_out["opportunity_list"] = [o.model_dump() for o in ops]

        analyses.append(
            FeatureColumnAnalysis(
                column=col_name,
                detected_type=feature_type,
                characteristics=_characteristics_summary(feature_type, details),
                opportunity=opportunity,
                priority=priority,
                reason=reason,
                opportunities=ops,
                details=detail_out,
                is_potential_target=is_target,
                leakage_sensitive=leakage,
            )
        )

    relationships = detect_feature_relationships(df, type_map)
    # Exclude relationships that involve only targets incorrectly — keep height/weight etc.
    relationships = [
        r
        for r in relationships
        if not any(c in potential_targets and len(r.columns) == 1 for c in r.columns)
    ]
    return analyses, relationships, potential_targets


def generate_feature_engineering_report(
    df: pd.DataFrame,
    *,
    dataset_id: int,
    source: str,
) -> FeatureEngineeringOpportunityReport:
    """Build the full opportunity report without mutating ``df``."""
    # Integrity snapshot
    before_cols = list(df.columns)
    before_hash = pd.util.hash_pandas_object(df, index=True).sum()

    analyses, relationships, targets = generate_feature_engineering_opportunities(df)

    after_cols = list(df.columns)
    after_hash = pd.util.hash_pandas_object(df, index=True).sum()
    unchanged = before_cols == after_cols and before_hash == after_hash

    opportunities_detected = sum(
        1
        for a in analyses
        if a.opportunity
        and not a.opportunity.lower().startswith("none")
    ) + len(relationships)

    return FeatureEngineeringOpportunityReport(
        dataset_id=dataset_id,
        source=source,
        columns_analyzed=len(analyses),
        opportunities_detected=opportunities_detected,
        column_analyses=analyses,
        relationships=relationships,
        potential_targets=targets,
        transformations_applied=False,
        column_count_unchanged=unchanged and len(before_cols) == len(after_cols),
    )


def _resolve_analysis_frame(
    db: Session,
    dataset_id: int,
) -> tuple[pd.DataFrame, str, Path]:
    """Require Phase 4 cleaned checkpoint for Phase 5.1 opportunity analysis."""
    from app.services.pipeline_state import PipelineStateError, require_cleaned_dataframe

    try:
        _dataset, df, path = require_cleaned_dataframe(db, dataset_id)
    except PipelineStateError as exc:
        raise FeatureOpportunityError(str(exc)) from exc
    return df, "cleaned", path


def build_feature_opportunity_report(
    db: Session,
    dataset_id: int,
) -> FeatureEngineeringOpportunityReport:
    """Analyze opportunities on the Phase-4 cleaned dataset only."""
    df, source, path = _resolve_analysis_frame(db, dataset_id)
    original_bytes = path.read_bytes() if path.exists() else None

    report = generate_feature_engineering_report(
        df, dataset_id=dataset_id, source=source
    )

    if original_bytes is not None and path.read_bytes() != original_bytes:
        raise FeatureOpportunityError(
            "Integrity check failed: analysis unexpectedly modified the source file."
        )
    return report
