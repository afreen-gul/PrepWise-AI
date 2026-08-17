"""Data quality assessment (read-only issue detection).

Analyzes a persisted CSV and reports missing data, duplicates, invalid values,
outliers, and related issues. Does not modify the dataset.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.schemas.quality import (
    ClassDistributionEntry,
    ClassImbalanceReport,
    DataQualityReport,
    DuplicateColumnPair,
    PotentiallyRedundantColumnPair,
    DuplicateRowsIssue,
    InvalidValueIssue,
    MissingValueIssue,
    OutlierIssue,
    QualityScore,
    ScoreBreakdown,
    SuspiciousDataTypeIssue,
)
from app.services.dataset_profiler import (
    _is_datetime_like,
    _name_matches_target,
)
from app.services.dataset_service import DatasetServiceError, read_dataset_csv
from app.services.duplicate_columns import (
    detect_potentially_redundant_columns,
    find_exact_duplicate_column_groups,
    list_exact_duplicate_pairs,
)

HIGH_MISSING_THRESHOLD = 0.5
NUMERIC_AS_TEXT_RATIO = 0.8
DATE_AS_TEXT_RATIO = 0.8
MIN_NUMERIC_FOR_IQR = 8
SEVERE_IMBALANCE_MAJORITY = 0.9
MODERATE_IMBALANCE_MAJORITY = 0.8
MAX_TARGET_CLASSES = 50
MIN_TARGET_CLASSES = 2

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_AGE_NAME_PATTERN = re.compile(r"age", re.IGNORECASE)
_SALARY_NAME_PATTERN = re.compile(r"salary|wage|income|compensation", re.IGNORECASE)
_EMAIL_NAME_PATTERN = re.compile(r"email|e_mail|mail", re.IGNORECASE)
_DATE_NAME_PATTERN = re.compile(r"date|dob|birth|timestamp|time", re.IGNORECASE)


class DataQualityError(Exception):
    """Raised when quality assessment cannot be completed."""


def missing_severity(percentage: float) -> str:
    """Map missing percentage to Low, Medium, or High severity."""
    if percentage <= 0:
        return "Low"
    if percentage < 5:
        return "Low"
    if percentage <= 20:
        return "Medium"
    return "High"


def quality_level(score: int) -> str:
    """Map numeric score to Excellent / Good / Fair / Poor."""
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Poor"


def detect_missing_values(df: pd.DataFrame) -> list[MissingValueIssue]:
    """Report missing counts and severity per column."""
    n_rows = len(df)
    if n_rows == 0:
        return []

    issues: list[MissingValueIssue] = []
    for column in df.columns:
        null_count = int(df[column].isna().sum())
        if null_count == 0:
            continue
        percentage = (null_count / n_rows) * 100.0
        issues.append(
            MissingValueIssue(
                column_name=str(column),
                count=null_count,
                percentage=round(percentage, 2),
                severity=missing_severity(percentage),
            )
        )
    return sorted(issues, key=lambda item: item.percentage, reverse=True)


def detect_duplicate_rows(df: pd.DataFrame) -> DuplicateRowsIssue:
    """Count rows that are exact duplicates."""
    n_rows = len(df)
    count = int(df.duplicated().sum()) if n_rows else 0
    percentage = (count / n_rows * 100.0) if n_rows else 0.0
    return DuplicateRowsIssue(count=count, percentage=round(percentage, 2))


def detect_duplicate_columns(df: pd.DataFrame) -> list[DuplicateColumnPair]:
    """Columns with identical values (duplicate → original, first in column order)."""
    records = list_exact_duplicate_pairs(df)
    return [
        DuplicateColumnPair(
            column_a=r["original_column"],
            column_b=r["duplicate_column"],
            original_column=r["original_column"],
            duplicate_column=r["duplicate_column"],
            similarity=r["similarity"],
        )
        for r in records
    ]


def detect_potentially_redundant_column_pairs(
    df: pd.DataFrame,
) -> list[PotentiallyRedundantColumnPair]:
    duplicate_of = find_exact_duplicate_column_groups(df)
    near = detect_potentially_redundant_columns(
        df, exclude_columns=set(duplicate_of.keys())
    )
    return [
        PotentiallyRedundantColumnPair(
            column_a=item["column_a"],
            column_b=item["column_b"],
            similarity=item["similarity"],
        )
        for item in near
    ]


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    """Columns with at most one distinct non-null value."""
    return [
        str(column)
        for column in df.columns
        if df[column].nunique(dropna=True) <= 1
    ]


def detect_high_missing_columns(df: pd.DataFrame) -> list[str]:
    """Columns where more than half of the values are missing."""
    n_rows = len(df)
    if n_rows == 0:
        return []
    return [
        str(column)
        for column in df.columns
        if (df[column].isna().sum() / n_rows) > HIGH_MISSING_THRESHOLD
    ]


def _numeric_parse_ratio(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    coerced = pd.to_numeric(non_null.astype(str), errors="coerce")
    return float(coerced.notna().mean())


def detect_suspicious_data_types(df: pd.DataFrame) -> list[SuspiciousDataTypeIssue]:
    """Flag numeric or date content stored as text."""
    issues: list[SuspiciousDataTypeIssue] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue

        name = str(column)
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            numeric_ratio = _numeric_parse_ratio(series)
            if numeric_ratio >= NUMERIC_AS_TEXT_RATIO:
                issues.append(
                    SuspiciousDataTypeIssue(
                        column_name=name,
                        issue_type="numeric_stored_as_text",
                        description=(
                            f"{numeric_ratio:.0%} of non-null values parse as numbers."
                        ),
                    )
                )
                continue

            if _is_datetime_like(series) or _DATE_NAME_PATTERN.search(name):
                sample = series.dropna().head(200)
                if not sample.empty:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    if parsed.notna().mean() >= DATE_AS_TEXT_RATIO:
                        issues.append(
                            SuspiciousDataTypeIssue(
                                column_name=name,
                                issue_type="date_stored_as_text",
                                description="Values appear to be dates but are stored as text.",
                            )
                        )
    return issues


def _count_invalid_emails(series: pd.Series) -> int:
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return 0
    valid = non_null.str.match(_EMAIL_PATTERN)
    return int((~valid).sum())


def _count_negative_numeric(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce")
    return int((numeric < 0).sum())


def _count_impossible_dates(series: pd.Series) -> int:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    valid = parsed.notna()
    if not valid.any():
        return 0
    now = datetime.now(timezone.utc)
    years = parsed[valid].dt.year
    impossible = (years < 1900) | (years > now.year + 1)
    return int(impossible.sum())


def detect_invalid_values(df: pd.DataFrame) -> list[InvalidValueIssue]:
    """Heuristic invalid value checks by column name and type."""
    issues: list[InvalidValueIssue] = []

    for column in df.columns:
        name = str(column)
        series = df[column]

        if _AGE_NAME_PATTERN.search(name):
            count = _count_negative_numeric(series)
            if count:
                issues.append(
                    InvalidValueIssue(
                        column_name=name,
                        issue_type="negative_age",
                        count=count,
                        description="Age values should not be negative.",
                    )
                )

        if _SALARY_NAME_PATTERN.search(name):
            count = _count_negative_numeric(series)
            if count:
                issues.append(
                    InvalidValueIssue(
                        column_name=name,
                        issue_type="negative_salary",
                        count=count,
                        description="Salary or income values should not be negative.",
                    )
                )

        if _EMAIL_NAME_PATTERN.search(name):
            count = _count_invalid_emails(series)
            if count:
                issues.append(
                    InvalidValueIssue(
                        column_name=name,
                        issue_type="invalid_email",
                        count=count,
                        description="Values do not match a basic email format.",
                    )
                )

        if (
            _DATE_NAME_PATTERN.search(name)
            or pd.api.types.is_datetime64_any_dtype(series)
            or _is_datetime_like(series)
        ):
            count = _count_impossible_dates(series)
            if count:
                issues.append(
                    InvalidValueIssue(
                        column_name=name,
                        issue_type="impossible_date",
                        count=count,
                        description="Dates outside a plausible range (before 1900 or far future).",
                    )
                )

    return issues


def detect_outliers_iqr(df: pd.DataFrame) -> list[OutlierIssue]:
    """IQR-based outlier counts for numeric columns (report only)."""
    issues: list[OutlierIssue] = []
    for column in df.columns:
        series = pd.to_numeric(df[column], errors="coerce")
        clean = series.dropna()
        if len(clean) < MIN_NUMERIC_FOR_IQR:
            continue
        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = float(q1 - 1.5 * iqr)
        upper = float(q3 + 1.5 * iqr)
        mask = (clean < lower) | (clean > upper)
        count = int(mask.sum())
        if count:
            issues.append(
                OutlierIssue(
                    column_name=str(column),
                    outlier_count=count,
                    lower_bound=round(lower, 4),
                    upper_bound=round(upper, 4),
                )
            )
    return issues


def _pick_likely_target_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        name = str(column)
        if not _name_matches_target(name):
            continue
        nunique = df[column].nunique(dropna=True)
        if MIN_TARGET_CLASSES <= nunique <= MAX_TARGET_CLASSES:
            return name
    return None


def detect_class_imbalance(df: pd.DataFrame) -> ClassImbalanceReport:
    """Class distribution for a likely target column."""
    target = _pick_likely_target_column(df)
    if target is None:
        return ClassImbalanceReport(
            target_column=None,
            distribution=[],
            majority_percentage=None,
            imbalance_ratio=None,
            is_severe=False,
            message="No suitable target-like column found for imbalance analysis.",
        )

    counts = df[target].value_counts(dropna=True)
    total = int(counts.sum())
    if total == 0 or len(counts) < MIN_TARGET_CLASSES:
        return ClassImbalanceReport(
            target_column=target,
            distribution=[],
            majority_percentage=None,
            imbalance_ratio=None,
            is_severe=False,
            message="Target column does not have enough classes for imbalance analysis.",
        )

    distribution = [
        ClassDistributionEntry(
            label=str(label),
            count=int(count),
            percentage=round((count / total) * 100.0, 2),
        )
        for label, count in counts.items()
    ]
    majority_pct = max(entry.percentage for entry in distribution)
    min_count = counts.min()
    max_count = counts.max()
    ratio = float(max_count / min_count) if min_count else None
    is_severe = majority_pct >= (SEVERE_IMBALANCE_MAJORITY * 100)

    if is_severe:
        message = "Severe class imbalance detected — consider resampling or class weights."
    elif majority_pct >= (MODERATE_IMBALANCE_MAJORITY * 100):
        message = "Moderate class imbalance detected — review before modeling."
    else:
        message = "Class distribution appears relatively balanced."

    return ClassImbalanceReport(
        target_column=target,
        distribution=distribution,
        majority_percentage=round(majority_pct, 2),
        imbalance_ratio=round(ratio, 2) if ratio is not None else None,
        is_severe=is_severe,
        message=message,
    )


def compute_quality_score(
    *,
    n_rows: int,
    missing: list[MissingValueIssue],
    duplicate_rows: DuplicateRowsIssue,
    constant_columns: list[str],
    high_missing_columns: list[str],
    invalid_values: list[InvalidValueIssue],
    outliers: list[OutlierIssue],
    class_imbalance: ClassImbalanceReport,
) -> tuple[QualityScore, ScoreBreakdown]:
    """Derive a 0–100 score from detected issues (baseline 100)."""
    breakdown = ScoreBreakdown()
    score = 100.0

    # Missing values: severity-weighted, capped.
    missing_penalty = 0.0
    for issue in missing:
        if issue.severity == "Low":
            missing_penalty += 0.5
        elif issue.severity == "Medium":
            missing_penalty += 2.0
        else:
            missing_penalty += 4.0
    breakdown.missing_values = min(25.0, missing_penalty)
    score -= breakdown.missing_values

    if n_rows > 0:
        breakdown.duplicate_rows = min(15.0, duplicate_rows.percentage * 0.4)
    score -= breakdown.duplicate_rows

    breakdown.constant_columns = min(10.0, len(constant_columns) * 2.0)
    score -= breakdown.constant_columns

    breakdown.high_missing_columns = min(15.0, len(high_missing_columns) * 5.0)
    score -= breakdown.high_missing_columns

    invalid_penalty = 0.0
    for issue in invalid_values:
        invalid_penalty += min(3.0, 1.0 + (issue.count / max(n_rows, 1)) * 10.0)
    breakdown.invalid_values = min(15.0, invalid_penalty)
    score -= breakdown.invalid_values

    outlier_penalty = 0.0
    for issue in outliers:
        outlier_penalty += min(2.0, issue.outlier_count / max(n_rows, 1) * 20.0)
    breakdown.outliers = min(10.0, outlier_penalty)
    score -= breakdown.outliers

    if class_imbalance.target_column and class_imbalance.majority_percentage is not None:
        if class_imbalance.is_severe:
            breakdown.class_imbalance = 10.0
        elif class_imbalance.majority_percentage >= MODERATE_IMBALANCE_MAJORITY * 100:
            breakdown.class_imbalance = 5.0
    score -= breakdown.class_imbalance

    final_score = int(max(0, min(100, round(score))))
    return QualityScore(score=final_score, level=quality_level(final_score)), breakdown


def assess_dataframe(df: pd.DataFrame, dataset_id: int) -> DataQualityReport:
    """Build a full quality report without modifying ``df``."""
    missing = detect_missing_values(df)
    duplicate_rows = detect_duplicate_rows(df)
    duplicate_columns = detect_duplicate_columns(df)
    constant_columns = detect_constant_columns(df)
    high_missing_columns = detect_high_missing_columns(df)
    suspicious = detect_suspicious_data_types(df)
    invalid = detect_invalid_values(df)
    outliers = detect_outliers_iqr(df)
    imbalance = detect_class_imbalance(df)

    quality_score, breakdown = compute_quality_score(
        n_rows=len(df),
        missing=missing,
        duplicate_rows=duplicate_rows,
        constant_columns=constant_columns,
        high_missing_columns=high_missing_columns,
        invalid_values=invalid,
        outliers=outliers,
        class_imbalance=imbalance,
    )

    return DataQualityReport(
        dataset_id=dataset_id,
        quality_score=quality_score,
        score_breakdown=breakdown,
        missing_values=missing,
        duplicate_rows=duplicate_rows,
        duplicate_columns=duplicate_columns,
        potentially_redundant_columns=detect_potentially_redundant_column_pairs(df),
        constant_columns=constant_columns,
        high_missing_columns=high_missing_columns,
        suspicious_data_types=suspicious,
        invalid_values=invalid,
        outliers=outliers,
        class_imbalance=imbalance,
    )


def _get_dataset_row(db: Session, dataset_id: int) -> Dataset:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise DataQualityError(f"Dataset with id {dataset_id} was not found.")
    return dataset


def build_quality_report(db: Session, dataset_id: int) -> DataQualityReport:
    """Load dataset from disk and assess quality."""
    dataset = _get_dataset_row(db, dataset_id)
    path = Path(dataset.dataset_path)
    if not path.is_file():
        raise DataQualityError("Dataset file is missing on the server.")

    try:
        df = read_dataset_csv(path)
    except DatasetServiceError as exc:
        raise DataQualityError(str(exc)) from exc

    return assess_dataframe(df, dataset_id)
