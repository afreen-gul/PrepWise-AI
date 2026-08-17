"""Phase 5.2 — Datetime feature engineering.

Creates useful temporal features from datetime columns detected in Phase 5.1.
Does NOT modify the original upload or the Phase-4 cleaned file in place —
writes a separate featured copy under ``processed/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset
from app.schemas.feature_engineering import (
    DatasetShapeSnapshot,
    DatetimeColumnAnalysis,
    DatetimeFeatureEngineeringResult,
    DatetimeIssueFlag,
    GeneratedFeatureMeta,
    SkippedFeatureMeta,
)
from app.services.data_cleaner import DataCleanerError, get_cleaned_file_path
from app.services.dataset_service import DatasetServiceError, read_dataset_csv
from app.services.feature_opportunity_detector import (
    FeatureOpportunityError,
    _END_DATE_NAME,
    _START_DATE_NAME,
    detect_datetime_feature,
)

# Day-of-week convention (documented in metadata): Monday=0 … Sunday=6
DAYOFWEEK_CONVENTION = "Monday=0 … Sunday=6 (pandas dayofweek)"

_BIRTH_DATE_NAME = re.compile(
    r"birth|dob|date_of_birth|born",
    re.IGNORECASE,
)
_REFERENCE_DATE_NAME = re.compile(
    r"reference.?date|as_of|snapshot.?date|report.?date|event.?date|"
    r"^reference$|^as_of_date$",
    re.IGNORECASE,
)
_ORDER_DATE_NAME = re.compile(
    r"(order|purchase).*(date|time|at)|^(order_date|purchase_date)$",
    re.IGNORECASE,
)
_DELIVERY_DATE_NAME = re.compile(
    r"(deliver|arrival|shipped).*(date|time|at)|^(delivery_date|arrival_date)$",
    re.IGNORECASE,
)

PREVIEW_ROWS = 10


class DatetimeFeatureEngineeringError(Exception):
    """Raised for expected datetime feature-engineering failures."""


_DATE_NAME_HINT = re.compile(
    r"date|time|timestamp|datetime|dob|birth|join|hire|start|end|"
    r"created|updated|expires|delivery|order|registration",
    re.IGNORECASE,
)


def is_datetime_column_for_engineering(series: pd.Series, column_name: str) -> bool:
    """Reuse Phase 5.1 detection; allow a minority of invalid values by name hint."""
    if detect_datetime_feature(series, column_name):
        return True
    if not _DATE_NAME_HINT.search(column_name):
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.head(min(200, len(non_null)))
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()) >= 0.5


def analyze_datetime_column(
    parsed: pd.Series,
    column_name: str,
    *,
    invalid_count: int,
) -> DatetimeColumnAnalysis:
    """Compute min/max/unique/range and whether time components vary."""
    valid = parsed.dropna()
    if valid.empty:
        return DatetimeColumnAnalysis(
            column=column_name,
            unique_dates=0,
            contains_time=False,
            invalid_count=invalid_count,
            parseable_count=0,
        )

    min_ts = valid.min()
    max_ts = valid.max()
    # Unique calendar dates
    unique_dates = int(valid.dt.normalize().nunique())
    range_days = float((max_ts - min_ts).total_seconds() / 86400.0)

    time_parts = valid.dt.hour.astype(int) + valid.dt.minute.astype(int) + valid.dt.second.astype(int)
    contains_time = bool((time_parts != 0).any())

    return DatetimeColumnAnalysis(
        column=column_name,
        min_date=str(min_ts.date()) if hasattr(min_ts, "date") else str(min_ts),
        max_date=str(max_ts.date()) if hasattr(max_ts, "date") else str(max_ts),
        unique_dates=unique_dates,
        date_range_days=round(range_days, 2),
        contains_time=contains_time,
        invalid_count=invalid_count,
        parseable_count=int(len(valid)),
    )


def parse_datetime_series(series: pd.Series) -> tuple[pd.Series, int]:
    """Parse once; return datetime64 series and count of unparseable non-nulls."""
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        invalid = int(series.notna().sum() - parsed.notna().sum())
        return parsed, invalid

    non_null_mask = series.notna() & series.astype(str).str.strip().ne("")
    parsed = pd.to_datetime(series, errors="coerce")
    invalid = int((non_null_mask & parsed.isna()).sum())
    return parsed, invalid


def _has_variation(values: pd.Series) -> bool:
    return int(values.dropna().nunique()) > 1


def _try_add_feature(
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
    selected: set[str] | None = None,
) -> None:
    """Add a feature if it does not exist, is selected, and is non-constant."""
    if selected is not None and feature_name not in selected:
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=source,
                reason="Not selected by user.",
                category="datetime",
            )
        )
        return
    if feature_name in working.columns:
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=source,
                reason="Feature already exists — skipped.",
            )
        )
        return

    if not _has_variation(values):
        skipped.append(
            SkippedFeatureMeta(
                feature=feature_name,
                source=source,
                reason="Skipped because the candidate feature has no variation (constant).",
            )
        )
        return

    working[feature_name] = values
    generated.append(
        GeneratedFeatureMeta(
            feature=feature_name,
            source=source,
            feature_type=feature_type,
            category="datetime",
            transformation=transformation,
            reason=reason,
            rows_affected=int(values.notna().sum()),
            status="Created",
        )
    )


def engineer_components_for_column(
    working: pd.DataFrame,
    column: str,
    parsed: pd.Series,
    analysis: DatetimeColumnAnalysis,
    generated: list[GeneratedFeatureMeta],
    skipped: list[SkippedFeatureMeta],
    selected: set[str] | None = None,
) -> None:
    """Create Year/Month/Day/DayOfWeek/IsWeekend/Hour/Minute when useful."""
    prefix = column

    # Year
    year = parsed.dt.year
    if _has_variation(year):
        _try_add_feature(
            working,
            feature_name=f"{prefix}_Year",
            values=year.astype("Int64"),
            source=column,
            transformation="Datetime → Year",
            reason="Year captures multi-year temporal patterns.",
            feature_type="Integer",
            generated=generated,
            skipped=skipped,
            selected=selected,
        )
    else:
        skipped.append(
            SkippedFeatureMeta(
                feature=f"{prefix}_Year",
                source=column,
                reason="Year feature skipped: all observations belong to the same year.",
            )
        )

    # Month
    month = parsed.dt.month
    if _has_variation(month):
        _try_add_feature(
            working,
            feature_name=f"{prefix}_Month",
            values=month.astype("Int64"),
            source=column,
            transformation="Datetime → Month",
            reason="Month information may capture seasonal or temporal patterns.",
            feature_type="Integer",
            generated=generated,
            skipped=skipped,
            selected=selected,
        )
    else:
        skipped.append(
            SkippedFeatureMeta(
                feature=f"{prefix}_Month",
                source=column,
                reason="Month feature skipped: all observations occur in the same month.",
            )
        )

    # Day of month
    day = parsed.dt.day
    if _has_variation(day):
        _try_add_feature(
            working,
            feature_name=f"{prefix}_Day",
            values=day.astype("Int64"),
            source=column,
            transformation="Datetime → Day of month",
            reason="Day-of-month variation may capture within-month patterns.",
            feature_type="Integer",
            generated=generated,
            skipped=skipped,
            selected=selected,
        )
    else:
        skipped.append(
            SkippedFeatureMeta(
                feature=f"{prefix}_Day",
                source=column,
                reason="Day feature skipped: no day-of-month variation.",
            )
        )

    # Day of week — Monday=0 … Sunday=6
    dow = parsed.dt.dayofweek
    if _has_variation(dow):
        _try_add_feature(
            working,
            feature_name=f"{prefix}_DayOfWeek",
            values=dow.astype("Int64"),
            source=column,
            transformation=f"Datetime → DayOfWeek ({DAYOFWEEK_CONVENTION})",
            reason=(
                f"Day of week may capture weekly patterns. Convention: {DAYOFWEEK_CONVENTION}."
            ),
            feature_type="Integer",
            generated=generated,
            skipped=skipped,
            selected=selected,
        )
    else:
        skipped.append(
            SkippedFeatureMeta(
                feature=f"{prefix}_DayOfWeek",
                source=column,
                reason="DayOfWeek feature skipped: all dates fall on the same weekday.",
            )
        )

    # Weekend
    is_weekend = (dow >= 5).astype("Int64")
    # Preserve NA where date missing
    is_weekend = is_weekend.where(parsed.notna(), other=pd.NA)
    if _has_variation(is_weekend):
        _try_add_feature(
            working,
            feature_name=f"{prefix}_IsWeekend",
            values=is_weekend,
            source=column,
            transformation="Datetime → IsWeekend (0=weekday, 1=weekend)",
            reason="Weekend indicator may capture weekend vs weekday behavior.",
            feature_type="Integer",
            generated=generated,
            skipped=skipped,
            selected=selected,
        )
    else:
        valid_dow = dow.dropna()
        if not valid_dow.empty and (valid_dow < 5).all():
            reason = "All observations occur on weekdays."
        elif not valid_dow.empty and (valid_dow >= 5).all():
            reason = "All observations occur on weekends."
        else:
            reason = "IsWeekend has no variation."
        skipped.append(
            SkippedFeatureMeta(
                feature=f"{prefix}_IsWeekend",
                source=column,
                reason=f"Skipped because {reason}",
            )
        )

    # Time components only when meaningful time exists
    if not analysis.contains_time:
        for part in ("Hour", "Minute", "Second"):
            skipped.append(
                SkippedFeatureMeta(
                    feature=f"{prefix}_{part}",
                    source=column,
                    reason=(
                        "Time components skipped: timestamps have no meaningful "
                        "time variation (e.g. all 00:00:00)."
                    ),
                )
            )
        return

    hour = parsed.dt.hour
    if _has_variation(hour):
        _try_add_feature(
            working,
            feature_name=f"{prefix}_Hour",
            values=hour.astype("Int64"),
            source=column,
            transformation="Datetime → Hour",
            reason="Hour captures within-day temporal patterns.",
            feature_type="Integer",
            generated=generated,
            skipped=skipped,
            selected=selected,
        )
    else:
        skipped.append(
            SkippedFeatureMeta(
                feature=f"{prefix}_Hour",
                source=column,
                reason="Hour feature skipped: no hour variation.",
            )
        )

    minute = parsed.dt.minute
    if _has_variation(minute):
        _try_add_feature(
            working,
            feature_name=f"{prefix}_Minute",
            values=minute.astype("Int64"),
            source=column,
            transformation="Datetime → Minute",
            reason="Minute variation detected in timestamps.",
            feature_type="Integer",
            generated=generated,
            skipped=skipped,
            selected=selected,
        )
    else:
        skipped.append(
            SkippedFeatureMeta(
                feature=f"{prefix}_Minute",
                source=column,
                reason="Minute feature skipped: no minute variation.",
            )
        )

    # Never auto-generate seconds
    skipped.append(
        SkippedFeatureMeta(
            feature=f"{prefix}_Second",
            source=column,
            reason="Seconds are not auto-generated (low utility).",
        )
    )


def _meaningful_datetime_pairs(datetime_cols: list[str]) -> list[tuple[str, str, str]]:
    """Return (start, end, kind) pairs using name heuristics only."""
    pairs: list[tuple[str, str, str]] = []
    starts = [c for c in datetime_cols if _START_DATE_NAME.search(c)]
    ends = [c for c in datetime_cols if _END_DATE_NAME.search(c)]
    for s in starts:
        for e in ends:
            if s != e:
                pairs.append((s, e, "duration"))

    orders = [c for c in datetime_cols if _ORDER_DATE_NAME.search(c)]
    deliveries = [c for c in datetime_cols if _DELIVERY_DATE_NAME.search(c)]
    for o in orders:
        for d in deliveries:
            if o != d:
                pairs.append((o, d, "duration"))

    births = [c for c in datetime_cols if _BIRTH_DATE_NAME.search(c)]
    refs = [c for c in datetime_cols if _REFERENCE_DATE_NAME.search(c)]
    for b in births:
        for r in refs:
            if b != r:
                pairs.append((b, r, "age"))

    # Deduplicate while preserving order
    seen: set[tuple[str, str, str]] = set()
    unique: list[tuple[str, str, str]] = []
    for item in pairs:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def engineer_duration_and_age(
    working: pd.DataFrame,
    parsed_map: dict[str, pd.Series],
    datetime_cols: list[str],
    generated: list[GeneratedFeatureMeta],
    skipped: list[SkippedFeatureMeta],
    issues: list[DatetimeIssueFlag],
    selected: set[str] | None = None,
) -> None:
    """Create Duration_Days / Age_Years for meaningful pairs only."""
    for left, right, kind in _meaningful_datetime_pairs(datetime_cols):
        left_ts = parsed_map[left]
        right_ts = parsed_map[right]
        delta_days = (right_ts - left_ts).dt.total_seconds() / 86400.0

        if kind == "duration":
            feature_name = f"{left}_{right}_Duration_Days"
            values = delta_days.round(2)
            # Flag negative durations (end before start)
            neg = int((delta_days.dropna() < 0).sum())
            if neg:
                issues.append(
                    DatetimeIssueFlag(
                        issue="negative_duration",
                        columns=[left, right],
                        count=neg,
                        message=(
                            f"{neg} row(s) have {right} before {left}; "
                            "duration may be negative. Dates were not modified."
                        ),
                    )
                )
            _try_add_feature(
                working,
                feature_name=feature_name,
                values=values,
                source=f"{left}, {right}",
                transformation="Datetime difference → Duration_Days",
                reason=f"Duration in days between '{left}' and '{right}'.",
                feature_type="Float",
                generated=generated,
                skipped=skipped,
                selected=selected,
            )
        else:
            # Age in whole years (floor)
            feature_name = f"{left}_{right}_Age_Years"
            age_float = delta_days / 365.25
            age_years = age_float.apply(
                lambda x: pd.NA if pd.isna(x) else int(x // 1)
            ).astype("Int64")
            # Birth after reference
            future = int((delta_days.dropna() < 0).sum())
            if future:
                issues.append(
                    DatetimeIssueFlag(
                        issue="birth_after_reference",
                        columns=[left, right],
                        count=future,
                        message=(
                            f"{future} row(s) have {left} after {right}; "
                            "flagged as inconsistent. Dates were not modified."
                        ),
                    )
                )
            _try_add_feature(
                working,
                feature_name=feature_name,
                values=age_years,
                source=f"{left}, {right}",
                transformation="Birth/reference dates → Age_Years",
                reason=(
                    f"Age in whole years from '{left}' relative to '{right}' "
                    "(no blind use of today's date)."
                ),
                feature_type="Integer",
                generated=generated,
                skipped=skipped,
                selected=selected,
            )


def engineer_datetime_features(
    df: pd.DataFrame,
    *,
    selected: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply Phase 5.2 datetime FE on a copy of ``df``.

    ``selected`` None = create all viable features; otherwise only those names.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise DatetimeFeatureEngineeringError("Dataset could not be loaded.")

    working = df.copy()
    before_rows = len(working)
    before_cols = list(working.columns)

    generated: list[GeneratedFeatureMeta] = []
    skipped: list[SkippedFeatureMeta] = []
    issues: list[DatetimeIssueFlag] = []
    analyses: list[DatetimeColumnAnalysis] = []
    parsed_map: dict[str, pd.Series] = {}

    datetime_cols: list[str] = []
    for column in working.columns:
        col_name = str(column)
        if not is_datetime_column_for_engineering(working[column], col_name):
            continue
        datetime_cols.append(col_name)

    for col_name in datetime_cols:
        parsed, invalid = parse_datetime_series(working[col_name])
        parsed_map[col_name] = parsed
        analysis = analyze_datetime_column(parsed, col_name, invalid_count=invalid)
        analyses.append(analysis)
        if invalid:
            issues.append(
                DatetimeIssueFlag(
                    issue="invalid_datetime",
                    columns=[col_name],
                    count=invalid,
                    message=(
                        f"{invalid} value(s) in '{col_name}' could not be parsed; "
                        "derived features are missing for those rows. "
                        "No rows were dropped."
                    ),
                )
            )
        engineer_components_for_column(
            working, col_name, parsed, analysis, generated, skipped, selected=selected
        )

    engineer_duration_and_age(
        working, parsed_map, datetime_cols, generated, skipped, issues, selected=selected
    )

    after_rows = len(working)
    if after_rows != before_rows:
        raise DatetimeFeatureEngineeringError(
            "Row count changed during datetime feature engineering — aborting."
        )

    # Original datetime columns must still exist
    for col_name in datetime_cols:
        if col_name not in working.columns:
            raise DatetimeFeatureEngineeringError(
                f"Original datetime column '{col_name}' was removed unexpectedly."
            )

    details = {
        "datetime_columns": datetime_cols,
        "analyses": analyses,
        "generated": generated,
        "skipped": skipped,
        "issues": issues,
        "before_columns": before_cols,
        "before_rows": before_rows,
        "after_rows": after_rows,
    }
    return working, details


def _featured_output_path(original_filename: str) -> Path:
    from app.services.pipeline_state import feature_engineered_dataset_path

    return feature_engineered_dataset_path(original_filename)


def _build_preview(df: pd.DataFrame) -> list[dict[str, Any]]:
    head = df.head(PREVIEW_ROWS).where(pd.notnull(df.head(PREVIEW_ROWS)), None)
    return head.to_dict(orient="records")


def apply_datetime_feature_engineering(
    db: Session,
    dataset_id: int,
) -> DatetimeFeatureEngineeringResult:
    """Engineer datetime features from the Phase-4 cleaned checkpoint only."""
    from app.services.pipeline_state import PipelineStateError, require_cleaned_dataframe

    try:
        dataset, df, cleaned_path = require_cleaned_dataframe(db, dataset_id)
    except PipelineStateError as exc:
        raise DatetimeFeatureEngineeringError(str(exc)) from exc

    original_upload = Path(dataset.dataset_path)
    original_bytes = original_upload.read_bytes() if original_upload.exists() else None
    cleaned_bytes = cleaned_path.read_bytes()

    before = DatasetShapeSnapshot(rows=len(df), columns=len(df.columns))
    engineered, details = engineer_datetime_features(df)
    after = DatasetShapeSnapshot(rows=len(engineered), columns=len(engineered.columns))

    destination = _featured_output_path(dataset.filename)
    try:
        engineered.to_csv(destination, index=False)
    except Exception as exc:
        raise DatetimeFeatureEngineeringError(
            f"Failed to save featured dataset: {exc}"
        ) from exc

    if original_bytes is not None and original_upload.read_bytes() != original_bytes:
        raise DatetimeFeatureEngineeringError(
            "Safety check failed: original upload was modified."
        )
    if cleaned_path.read_bytes() != cleaned_bytes:
        raise DatetimeFeatureEngineeringError(
            "Safety check failed: cleaned checkpoint was modified."
        )

    return DatetimeFeatureEngineeringResult(
        dataset_id=dataset_id,
        source="cleaned",
        datetime_columns_analyzed=len(details["datetime_columns"]),
        features_generated=len(details["generated"]),
        features_skipped=len(details["skipped"]),
        before=before,
        after=after,
        new_features=len(details["generated"]),
        datetime_analyses=details["analyses"],
        generated_features=details["generated"],
        skipped_features=details["skipped"],
        issues=details["issues"],
        original_datetime_columns_preserved=True,
        row_count_unchanged=before.rows == after.rows,
        featured_filename=destination.name,
        featured_path=str(destination),
        preview=_build_preview(engineered),
        download_url=f"/api/v1/datasets/{dataset_id}/feature-engineering/datetime/download",
        message=(
            "Datetime feature engineering applied to the cleaned dataset. "
            "Original upload and cleaned checkpoint were not modified."
        ),
    )


def get_featured_file_path(db: Session, dataset_id: int) -> Path:
    """Resolve the featured CSV path for a dataset."""
    from app.services.pipeline_state import (
        feature_engineered_dataset_path,
        get_dataset_or_raise,
    )

    dataset = get_dataset_or_raise(db, dataset_id)
    path = feature_engineered_dataset_path(dataset.filename)
    if not path.is_file():
        raise DatetimeFeatureEngineeringError(
            "No featured dataset found. Apply datetime feature engineering first."
        )
    return path
