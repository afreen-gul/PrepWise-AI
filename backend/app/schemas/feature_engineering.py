"""Pydantic schemas for Phase 5.1 feature-engineering opportunity detection.

Analysis / recommendation only — no transformations are performed.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeatureOpportunity(BaseModel):
    """One recommended (not applied) feature-engineering opportunity."""

    opportunity: str
    priority: str = Field(description="HIGH | MEDIUM | LOW")
    reason: str


class FeatureColumnAnalysis(BaseModel):
    """Per-column type detection and opportunity summary."""

    column: str
    detected_type: str
    characteristics: str
    opportunity: str
    priority: str
    reason: str
    opportunities: list[FeatureOpportunity] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    is_potential_target: bool = False
    leakage_sensitive: bool = False


class FeatureRelationshipOpportunity(BaseModel):
    """Suggested relationship between columns (not materialized)."""

    columns: list[str]
    opportunity: str
    priority: str
    reason: str


class FeatureEngineeringOpportunityReport(BaseModel):
    """Full Phase 5.1 opportunity report (read-only)."""

    dataset_id: int
    source: str = Field(
        description="cleaned | original — which file was analyzed"
    )
    columns_analyzed: int
    opportunities_detected: int
    column_analyses: list[FeatureColumnAnalysis]
    relationships: list[FeatureRelationshipOpportunity] = Field(
        default_factory=list
    )
    potential_targets: list[str] = Field(default_factory=list)
    transformations_applied: bool = False
    column_count_unchanged: bool = True
    message: str = (
        "Phase 5.1 analysis only — no feature transformations were applied."
    )


class FeatureCandidate(BaseModel):
    """One concrete engineered feature the user may opt into creating."""

    id: str = Field(description="Stable id = proposed feature column name")
    feature: str
    source: str
    category: str = Field(description="datetime | numerical | text")
    transformation: str
    reason: str
    priority: str = "MEDIUM"
    feature_type: str = "Integer"
    default_selected: bool = True


class FeatureEngineeringCandidateReport(BaseModel):
    """Phase 5 recommendation report — no features are created yet."""

    dataset_id: int
    source: str = "cleaned"
    columns_analyzed: int
    candidates: list[FeatureCandidate] = Field(default_factory=list)
    candidates_count: int = 0
    engineering_recommended: bool = False
    potential_targets: list[str] = Field(default_factory=list)
    polynomial_recommendations: list[str] = Field(default_factory=list)
    transformations_applied: bool = False
    message: str = (
        "Feature engineering recommendations only — nothing was created yet. "
        "Select candidates, then generate."
    )


class FeatureEngineeringApplyRequest(BaseModel):
    """Generate only the user-selected engineered features.

    - selected_feature_ids is None → generate all viable candidates (legacy/tests)
    - selected_feature_ids is [] → pass-through cleaned columns (no new features)
    - otherwise → generate only the listed feature names
    """

    selected_feature_ids: list[str] | None = None


# ---------------------------------------------------------------------------
# Phase 5.2 — Datetime feature engineering (creates features)
# ---------------------------------------------------------------------------


class DatetimeColumnAnalysis(BaseModel):
    """Summary stats for one datetime source column."""

    column: str
    min_date: str | None = None
    max_date: str | None = None
    unique_dates: int = 0
    date_range_days: float | None = None
    contains_time: bool = False
    invalid_count: int = 0
    parseable_count: int = 0


class GeneratedFeatureMeta(BaseModel):
    """Metadata for one newly created engineered feature."""

    feature: str
    source: str
    feature_type: str = "Integer"
    category: str = "datetime"  # datetime | numerical | text
    transformation: str
    reason: str
    rows_affected: int
    status: str = "Created"
    phase: int = 5
    before_stats: dict[str, Any] | None = None
    after_stats: dict[str, Any] | None = None


class PipelinePhaseStatus(BaseModel):
    phase: int
    name: str
    status: str
    requires: str | None = None
    output: str | None = None
    path: str | None = None
    metadata_path: str | None = None
    note: str | None = None


class PipelineStatusResponse(BaseModel):
    """Cumulative PrepWise pipeline checkpoint status."""

    dataset_id: int
    filename: str
    current_stage: str
    current_stage_label: str
    raw_immutable: bool = True
    raw_path: str
    cleaned_available: bool
    feature_engineered_available: bool
    feature_selected_available: bool = False
    phase5_ready: bool
    phase6_ready: bool
    phase7_ready: bool = False
    exports: dict[str, str | None] = Field(default_factory=dict)
    phases: list[PipelinePhaseStatus] = Field(default_factory=list)
    message: str = ""


class SkippedFeatureMeta(BaseModel):
    """Record of a feature that was considered but not created."""

    feature: str
    source: str
    status: str = "Skipped"
    reason: str
    category: str | None = None


class RemovedFeatureMeta(BaseModel):
    """Generated feature removed during Phase 5.5 validation."""

    feature: str
    source: str | None = None
    category: str | None = None
    status: str = "Removed"
    reason: str


class DatetimeIssueFlag(BaseModel):
    """Data-quality note related to datetime engineering (non-destructive)."""

    issue: str
    columns: list[str]
    count: int
    message: str


class DatasetShapeSnapshot(BaseModel):
    rows: int
    columns: int


class DatetimeFeatureEngineeringResult(BaseModel):
    """Phase 5.2 apply result — new columns added; originals preserved."""

    dataset_id: int
    source: str
    datetime_columns_analyzed: int
    features_generated: int
    features_skipped: int
    before: DatasetShapeSnapshot
    after: DatasetShapeSnapshot
    new_features: int
    datetime_analyses: list[DatetimeColumnAnalysis] = Field(default_factory=list)
    generated_features: list[GeneratedFeatureMeta] = Field(default_factory=list)
    skipped_features: list[SkippedFeatureMeta] = Field(default_factory=list)
    issues: list[DatetimeIssueFlag] = Field(default_factory=list)
    original_datetime_columns_preserved: bool = True
    row_count_unchanged: bool = True
    featured_filename: str | None = None
    featured_path: str | None = None
    preview: list[dict[str, Any]] = Field(default_factory=list)
    download_url: str | None = None
    message: str = (
        "Phase 5.2 datetime feature engineering applied to a working copy. "
        "Original upload and cleaned files were not modified."
    )


class Phase5FeatureEngineeringResult(BaseModel):
    """Full Phase 5 apply result (5.2–5.5)."""

    dataset_id: int
    source: str
    pipeline_stage_before: str = "cleaned"
    pipeline_stage_after: str = "feature_engineered"
    applied_to_cleaned_dataset: bool = True
    selected_feature_ids: list[str] | None = None
    selection_mode: str = Field(
        default="all",
        description="all | selected | pass_through",
    )
    original_feature_count: int
    datetime_features_generated: int = 0
    numerical_features_generated: int = 0
    text_features_generated: int = 0
    features_generated: int = 0
    features_skipped: int = 0
    features_removed: int = 0
    final_feature_count: int = 0
    before: DatasetShapeSnapshot
    after: DatasetShapeSnapshot
    generated_features: list[GeneratedFeatureMeta] = Field(default_factory=list)
    skipped_features: list[SkippedFeatureMeta] = Field(default_factory=list)
    removed_features: list[RemovedFeatureMeta] = Field(default_factory=list)
    issues: list[DatetimeIssueFlag] = Field(default_factory=list)
    datetime_analyses: list[DatetimeColumnAnalysis] = Field(default_factory=list)
    polynomial_recommendations: list[str] = Field(default_factory=list)
    original_columns_preserved: bool = True
    row_count_unchanged: bool = True
    featured_filename: str | None = None
    featured_path: str | None = None
    metadata_path: str | None = None
    preview: list[dict[str, Any]] = Field(default_factory=list)
    download_url: str | None = None
    message: str = (
        "Feature engineering applied to the cleaned dataset. "
        "Original upload and cleaned checkpoint were not modified."
    )
