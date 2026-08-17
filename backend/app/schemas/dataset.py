"""Pydantic schemas for dataset request/response serialization.

These define the shape of data crossing the API boundary and keep the ORM
models decoupled from the public contract.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DatasetBase(BaseModel):
    """Fields shared across dataset schemas."""

    filename: str
    rows: int
    columns: int
    file_size: int
    dataset_path: str


class DatasetRead(DatasetBase):
    """Dataset metadata returned to clients (persisted representation)."""

    id: int
    upload_date: datetime

    # Allow building this schema directly from an ORM object.
    model_config = ConfigDict(from_attributes=True)


class DatasetUploadResponse(BaseModel):
    """Full response returned after a successful upload.

    Combines persisted metadata with a lightweight preview so the frontend can
    render an overview without a second request.
    """

    dataset: DatasetRead
    column_names: list[str]
    preview: list[dict]  # first N rows, each row as {column: value}
