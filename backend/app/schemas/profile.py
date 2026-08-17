"""Pydantic schemas for dataset profiling responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetShape(BaseModel):
    """Row and column counts."""

    rows: int
    columns: int


class ColumnSummary(BaseModel):
    """Per-column statistics for profiling."""

    name: str
    data_type: str
    non_null_count: int
    null_count: int
    unique_values: int
    example_value: str | None
    semantic_type: str = Field(
        description="Inferred role: numerical, categorical, datetime, boolean, or text."
    )


class IdentifierColumn(BaseModel):
    """Column likely used as a record identifier."""

    column_name: str
    reason: str
    recommendation: str = "Ignore during model training."


class PotentialTargetColumn(BaseModel):
    """Column that may be a prediction target (recommendation only)."""

    column_name: str
    reason: str


class HighCardinalityColumn(BaseModel):
    """Column with many distinct values relative to dataset size."""

    column_name: str
    unique_count: int
    recommendation: str


class ColumnTypeGroups(BaseModel):
    """Columns grouped by inferred semantic type."""

    numerical: list[str]
    categorical: list[str]
    datetime: list[str]
    boolean: list[str]
    text: list[str]


class DatasetProfile(BaseModel):
    """Complete read-only profile for an uploaded dataset."""

    dataset_id: int
    shape: DatasetShape
    memory_usage_bytes: int
    column_summaries: list[ColumnSummary]
    column_types: ColumnTypeGroups
    identifier_columns: list[IdentifierColumn]
    potential_target_columns: list[PotentialTargetColumn]
    constant_columns: list[str]
    high_cardinality_columns: list[HighCardinalityColumn]
