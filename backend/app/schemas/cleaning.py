"""Pydantic schemas for intelligent data cleaning (Phase 4)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OutlierStrategy(str, Enum):
    """Configurable outlier handling strategies."""

    FLAG = "flag"
    REMOVE = "remove"
    CLIP = "clip"


class CleaningConfig(BaseModel):
    """User-selected cleaning options."""

    remove_duplicate_rows: bool = True
    remove_duplicate_columns: bool = Field(
        default=True,
        description="Remove columns that are exact value duplicates of an earlier column.",
    )
    convert_safe_dtypes: bool = True
    handle_missing_values: bool = True
    remove_constant_columns: bool = False
    drop_high_missing_columns: bool = False
    handle_invalid_values: bool = True
    handle_empty_strings: bool = True
    handle_outliers: bool = True
    outlier_strategy: OutlierStrategy = OutlierStrategy.FLAG
    high_missingness_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Fraction of missing values above which a column is high-missing.",
    )
    # Categorical missing-value imputation (Phase 4)
    min_group_size: int = Field(default=10, ge=1)
    min_group_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_grouping_cardinality: int = Field(default=50, ge=2)
    global_mode_min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    min_valid_observations: int = Field(default=20, ge=1)
    min_unique_categories: int = Field(default=2, ge=2)
    categorical_random_state: int = Field(default=42)


class CleaningLogEntry(BaseModel):
    """One recorded transformation or review decision."""

    operation: str
    column: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class IssueTransition(BaseModel):
    """Before → after counts for a cleaning summary line."""

    issue: str
    before: int | float | str
    after: int | float | str
    action: str


class CleaningSummary(BaseModel):
    """Dry-run / apply summary of what will be fixed vs flagged."""

    issues_found: list[str]
    issues_to_fix: list[IssueTransition]
    issues_to_flag: list[IssueTransition]


class DatasetSnapshot(BaseModel):
    """Lightweight before/after dataset metrics."""

    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    outliers_detected: int
    columns_modified: list[str]


class CleaningPreviewResponse(BaseModel):
    """Dry-run response — no files written."""

    dataset_id: int
    config: CleaningConfig
    summary: CleaningSummary
    before: DatasetSnapshot
    planned_log: list[CleaningLogEntry]


class CleaningApplyResponse(BaseModel):
    """Result after cleaning is applied and a cleaned copy is saved."""

    dataset_id: int
    config: CleaningConfig
    summary: CleaningSummary
    before: DatasetSnapshot
    after: DatasetSnapshot
    cleaning_log: list[CleaningLogEntry]
    cleaned_filename: str
    cleaned_path: str
    preview: list[dict[str, Any]]
    download_url: str
