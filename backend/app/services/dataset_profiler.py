"""Intelligent dataset profiling (read-only analysis).

Loads a persisted CSV and produces structural metadata, column summaries, and
heuristic recommendations. Does not modify the dataset.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.schemas.profile import (
    ColumnSummary,
    ColumnTypeGroups,
    DatasetProfile,
    DatasetShape,
    HighCardinalityColumn,
    IdentifierColumn,
    PotentialTargetColumn,
)
from app.services.dataset_service import DatasetServiceError, read_dataset_csv

# Columns with more unique values than this are flagged (when non-numeric).
HIGH_CARDINALITY_UNIQUE_THRESHOLD = 50

# Share of rows above which a non-numeric column is high-cardinality.
HIGH_CARDINALITY_RATIO_THRESHOLD = 0.5

# Object columns with at most this many uniques are treated as categorical.
CATEGORICAL_UNIQUE_MAX = 50

# Minimum share of parseable values to classify a column as datetime.
DATETIME_PARSE_RATIO = 0.8

# Sample size for datetime inference on object columns.
DATETIME_SAMPLE_SIZE = 200

_IDENTIFIER_NAME_PATTERN = re.compile(
    r"(^id$|_id$|_id_|^uuid$|uuid|guid|invoice|customer.?id|employee.?id|"
    r"user.?id|record.?id|transaction.?id)",
    re.IGNORECASE,
)

_TARGET_NAME_PATTERN = re.compile(
    r"(target|label|class|outcome|churn|purchased|purchase|fraud|survived|"
    r"default|approved|success|failure|response|y_true|y_label)",
    re.IGNORECASE,
)

_BOOLEAN_LITERALS = {
    "true",
    "false",
    "yes",
    "no",
    "y",
    "n",
    "0",
    "1",
    "t",
    "f",
}


class DatasetProfilerError(Exception):
    """Raised when profiling cannot be completed."""


def _truncate_example(value: object, max_len: int = 80) -> str | None:
    """Format a single example value for API output."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    if text.lower() == "nan":
        return None
    return text if len(text) <= max_len else f"{text[: max_len - 3]}..."


def example_value(series: pd.Series) -> str | None:
    """Return the first non-null value in a column."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    return _truncate_example(non_null.iloc[0])


def _is_boolean_like(series: pd.Series) -> bool:
    """Heuristic: column behaves like boolean flags."""
    if pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty or non_null.nunique() > 4:
        return False
    normalized = non_null.astype(str).str.strip().str.lower()
    return set(normalized.unique()).issubset(_BOOLEAN_LITERALS)


def _is_datetime_like(series: pd.Series) -> bool:
    """Heuristic: column stores datetimes (typed or parseable strings)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(
        series
    ):
        return False
    sample = series.dropna().head(DATETIME_SAMPLE_SIZE)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= DATETIME_PARSE_RATIO


def infer_semantic_type(series: pd.Series, column_name: str) -> str:
    """Classify a column into numerical, categorical, datetime, boolean, or text."""
    if _is_boolean_like(series):
        return "boolean"
    if _is_datetime_like(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    nunique = series.nunique(dropna=True)
    n_rows = len(series)
    if nunique <= CATEGORICAL_UNIQUE_MAX or (
        n_rows > 0 and nunique / n_rows <= 0.05
    ):
        return "categorical"
    return "text"


def summarize_column(series: pd.Series, column_name: str) -> ColumnSummary:
    """Build profiling statistics for one column."""
    non_null_count = int(series.notna().sum())
    null_count = int(series.isna().sum())
    unique_values = int(series.nunique(dropna=True))
    return ColumnSummary(
        name=column_name,
        data_type=str(series.dtype),
        non_null_count=non_null_count,
        null_count=null_count,
        unique_values=unique_values,
        example_value=example_value(series),
        semantic_type=infer_semantic_type(series, column_name),
    )


def _name_matches_identifier(column_name: str) -> bool:
    return bool(_IDENTIFIER_NAME_PATTERN.search(column_name.replace(" ", "_")))


def _name_matches_target(column_name: str) -> bool:
    return bool(_TARGET_NAME_PATTERN.search(column_name.replace(" ", "_")))


def detect_identifier_columns(
    df: pd.DataFrame,
    summaries: list[ColumnSummary],
) -> list[IdentifierColumn]:
    """Find columns that likely identify records (not features)."""
    n_rows = len(df)
    if n_rows == 0:
        return []

    results: list[IdentifierColumn] = []
    for summary in summaries:
        nunique = summary.unique_values
        reasons: list[str] = []

        if _name_matches_identifier(summary.name):
            reasons.append("Name suggests an identifier (e.g. ID or UUID).")
        if n_rows > 1 and nunique > 1 and (nunique / n_rows) >= 0.95:
            reasons.append(
                f"Very high uniqueness ({nunique:,} of {n_rows:,} rows are distinct)."
            )

        if reasons:
            results.append(
                IdentifierColumn(
                    column_name=summary.name,
                    reason=" ".join(reasons),
                )
            )
    return results


def detect_potential_target_columns(
    summaries: list[ColumnSummary],
) -> list[PotentialTargetColumn]:
    """Suggest columns that might be prediction targets (no auto-selection)."""
    results: list[PotentialTargetColumn] = []
    for summary in summaries:
        if _name_matches_target(summary.name):
            results.append(
                PotentialTargetColumn(
                    column_name=summary.name,
                    reason="Column name matches common target/label patterns.",
                )
            )
    return results


def detect_constant_columns(summaries: list[ColumnSummary]) -> list[str]:
    """Columns with at most one distinct non-null value."""
    return [s.name for s in summaries if s.unique_values <= 1]


def detect_high_cardinality_columns(
    df: pd.DataFrame,
    summaries: list[ColumnSummary],
) -> list[HighCardinalityColumn]:
    """Flag non-numeric columns with many distinct values."""
    n_rows = len(df)
    if n_rows == 0:
        return []

    results: list[HighCardinalityColumn] = []
    for summary in summaries:
        if summary.semantic_type in {"numerical", "boolean", "datetime"}:
            continue
        unique_count = summary.unique_values
        ratio = unique_count / n_rows if n_rows else 0.0
        if unique_count <= HIGH_CARDINALITY_UNIQUE_THRESHOLD and ratio <= (
            HIGH_CARDINALITY_RATIO_THRESHOLD
        ):
            continue
        results.append(
            HighCardinalityColumn(
                column_name=summary.name,
                unique_count=unique_count,
                recommendation=(
                    "High cardinality — consider grouping rare categories, "
                    "target encoding, or feature hashing before modeling."
                ),
            )
        )
    return results


def group_columns_by_type(summaries: list[ColumnSummary]) -> ColumnTypeGroups:
    """Aggregate column names by inferred semantic type."""
    groups: dict[str, list[str]] = {
        "numerical": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "text": [],
    }
    for summary in summaries:
        groups[summary.semantic_type].append(summary.name)
    return ColumnTypeGroups(**groups)


def profile_dataframe(df: pd.DataFrame, dataset_id: int) -> DatasetProfile:
    """Generate a full profile from an in-memory DataFrame (no mutations)."""
    summaries = [
        summarize_column(df[column], str(column)) for column in df.columns
    ]
    memory_bytes = int(df.memory_usage(deep=True).sum())

    return DatasetProfile(
        dataset_id=dataset_id,
        shape=DatasetShape(rows=int(df.shape[0]), columns=int(df.shape[1])),
        memory_usage_bytes=memory_bytes,
        column_summaries=summaries,
        column_types=group_columns_by_type(summaries),
        identifier_columns=detect_identifier_columns(df, summaries),
        potential_target_columns=detect_potential_target_columns(summaries),
        constant_columns=detect_constant_columns(summaries),
        high_cardinality_columns=detect_high_cardinality_columns(df, summaries),
    )


def _get_dataset_row(db: Session, dataset_id: int) -> Dataset:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise DatasetProfilerError(f"Dataset with id {dataset_id} was not found.")
    return dataset


def build_profile(db: Session, dataset_id: int) -> DatasetProfile:
    """Load a dataset from disk and return its profile."""
    dataset = _get_dataset_row(db, dataset_id)
    path = Path(dataset.dataset_path)
    if not path.is_file():
        raise DatasetProfilerError("Dataset file is missing on the server.")

    try:
        df = read_dataset_csv(path)
    except DatasetServiceError as exc:
        raise DatasetProfilerError(str(exc)) from exc

    return profile_dataframe(df, dataset_id)
