"""Pydantic schemas for data quality assessment responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QualityScore(BaseModel):
    """Overall dataset quality score and label."""

    score: int = Field(ge=0, le=100)
    level: str


class ScoreBreakdown(BaseModel):
    """Points deducted from the baseline score of 100."""

    missing_values: float = 0.0
    duplicate_rows: float = 0.0
    constant_columns: float = 0.0
    high_missing_columns: float = 0.0
    invalid_values: float = 0.0
    outliers: float = 0.0
    class_imbalance: float = 0.0


class MissingValueIssue(BaseModel):
    """Missing data in a single column."""

    column_name: str
    count: int
    percentage: float
    severity: str


class DuplicateRowsIssue(BaseModel):
    """Fully duplicated records."""

    count: int
    percentage: float


class DuplicateColumnPair(BaseModel):
    """Redundant column identical to an original (value-based detection)."""

    column_a: str
    column_b: str
    similarity: float = Field(default=100.0, ge=0.0, le=100.0)
    original_column: str | None = None
    duplicate_column: str | None = None


class PotentiallyRedundantColumnPair(BaseModel):
    """Highly similar columns that are not exact duplicates (report only)."""

    column_a: str
    column_b: str
    similarity: float = Field(ge=0.0, le=100.0)


class SuspiciousDataTypeIssue(BaseModel):
    """Column stored with an inappropriate dtype."""

    column_name: str
    issue_type: str
    description: str


class InvalidValueIssue(BaseModel):
    """Values that fail domain or format checks."""

    column_name: str
    issue_type: str
    count: int
    description: str


class OutlierIssue(BaseModel):
    """Outliers detected via the IQR method (report only)."""

    column_name: str
    outlier_count: int
    lower_bound: float
    upper_bound: float


class ClassDistributionEntry(BaseModel):
    """One class label in a target column."""

    label: str
    count: int
    percentage: float


class ClassImbalanceReport(BaseModel):
    """Distribution for a likely target column, if found."""

    target_column: str | None
    distribution: list[ClassDistributionEntry]
    majority_percentage: float | None
    imbalance_ratio: float | None
    is_severe: bool
    message: str


class DataQualityReport(BaseModel):
    """Complete read-only data quality assessment."""

    dataset_id: int
    quality_score: QualityScore
    score_breakdown: ScoreBreakdown
    missing_values: list[MissingValueIssue]
    duplicate_rows: DuplicateRowsIssue
    duplicate_columns: list[DuplicateColumnPair]
    potentially_redundant_columns: list[PotentiallyRedundantColumnPair] = Field(
        default_factory=list
    )
    constant_columns: list[str]
    high_missing_columns: list[str]
    suspicious_data_types: list[SuspiciousDataTypeIssue]
    invalid_values: list[InvalidValueIssue]
    outliers: list[OutlierIssue]
    class_imbalance: ClassImbalanceReport
