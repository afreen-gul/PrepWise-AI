"""Pydantic schemas for Phase 6 feature selection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.feature_engineering import DatasetShapeSnapshot


class FeatureSelectionAnalyzeRequest(BaseModel):
    """Optional target override for analysis."""

    target_column: str | None = Field(
        default=None,
        description="Explicit target column. If omitted, heuristic detection is used.",
    )


class FeatureSelectionApplyRequest(BaseModel):
    """Apply recommended selection with optional explicit REVIEW removals."""

    target_column: str | None = None
    apply_recommended: bool = Field(
        default=True,
        description="Exclude REMOVE recommendations when True.",
    )
    also_remove: list[str] = Field(
        default_factory=list,
        description="Extra features to remove (typically REVIEW features chosen by the user).",
    )
    force_keep: list[str] = Field(
        default_factory=list,
        description="Features to keep even if recommended REMOVE (except target still protected).",
    )


class FeatureQualityRow(BaseModel):
    feature: str
    datatype: str
    semantic_type: str
    missing_pct: float
    unique_count: int
    unique_pct: float
    most_frequent_value: str | None = None
    frequency_pct: float | None = None
    is_constant: bool = False
    is_near_constant: bool = False
    is_identifier: bool = False
    is_exact_duplicate: bool = False
    duplicate_of: str | None = None
    is_generated: bool = False
    source_feature: str | None = None
    transformation: str | None = None
    quality_flags: list[str] = Field(default_factory=list)


class CorrelationPairRow(BaseModel):
    feature_a: str
    feature_b: str
    correlation: float
    recommendation: str
    preferred_feature: str | None = None
    reason: str


class VIFRow(BaseModel):
    feature: str
    vif: float | None = None
    status: str
    related_features: list[str] = Field(default_factory=list)
    recommendation: str = "N/A"


class TargetScoreRow(BaseModel):
    feature: str
    target_type: str
    mi_score: float | None = None
    rank: int | None = None
    interpretation: str
    recommendation: str = "N/A"


class FeatureDecisionRow(BaseModel):
    feature: str
    feature_type: str
    status: str = Field(description="Quality / structural status summary")
    missing_pct: float | None = None
    unique_pct: float | None = None
    correlation: str | None = "N/A"
    vif: str | None = "N/A"
    target_score: str | None = "N/A"
    decision: str = Field(description="KEEP | REVIEW | REMOVE")
    reason: str
    evidence: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    is_target: bool = False
    is_generated: bool = False
    source_feature: str | None = None
    transformation: str | None = None


class FeatureSelectionSummary(BaseModel):
    total_features: int
    keep: int
    review: int
    remove: int
    target_column: str | None = None
    target_task: str | None = None
    target_aware_applied: bool = False


class FeatureSelectionReport(BaseModel):
    """Full Phase 6 analysis report (analyze or after apply)."""

    dataset_id: int
    source: str = "feature_engineered"
    pipeline_stage_before: str = "feature_engineered"
    summary: FeatureSelectionSummary
    quality_rows: list[FeatureQualityRow] = Field(default_factory=list)
    correlation_pairs: list[CorrelationPairRow] = Field(default_factory=list)
    vif_rows: list[VIFRow] = Field(default_factory=list)
    vif_available: bool = True
    vif_message: str | None = None
    target_scores: list[TargetScoreRow] = Field(default_factory=list)
    target_message: str | None = None
    decisions: list[FeatureDecisionRow] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)
    before: DatasetShapeSnapshot | None = None
    after: DatasetShapeSnapshot | None = None
    selected_filename: str | None = None
    selected_path: str | None = None
    report_path: str | None = None
    preview: list[dict[str, Any]] = Field(default_factory=list)
    download_url: str | None = None
    report_download_url: str | None = None
    applied: bool = False
    row_count_unchanged: bool = True
    feature_engineered_preserved: bool = True
    message: str = ""
