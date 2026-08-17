"""Business logic for dataset ingestion.

The service layer sits between the API routes and the database/filesystem. It
validates input, persists files, parses CSVs with pandas, and records metadata.
Routes stay thin and delegate all real work here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset
from app.utils import file_utils


class DatasetServiceError(Exception):
    """Raised for expected, user-facing dataset processing failures."""


# Number of preview rows exposed to the frontend.
PREVIEW_ROWS = 10


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame, translating errors to a clean message."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetServiceError("The uploaded CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetServiceError(
            "The uploaded file could not be parsed as valid CSV."
        ) from exc
    except UnicodeDecodeError as exc:
        raise DatasetServiceError(
            "The uploaded file is not valid UTF-8 encoded CSV."
        ) from exc


def _build_preview(df: pd.DataFrame, limit: int = PREVIEW_ROWS) -> list[dict]:
    """Return the first ``limit`` rows as JSON-serializable dictionaries.

    NaN values are converted to ``None`` so the payload is valid JSON.
    """
    head = df.head(limit).where(pd.notnull(df.head(limit)), None)
    return head.to_dict(orient="records")


def process_upload(
    db: Session,
    *,
    filename: str,
    content: bytes,
) -> tuple[Dataset, list[str], list[dict]]:
    """Validate, store, parse, and record an uploaded CSV.

    Returns the persisted ``Dataset`` row along with the column names and a
    preview of the first rows.

    Raises:
        DatasetServiceError: for validation or parsing problems.
    """
    # --- Validation --------------------------------------------------------
    if not filename:
        raise DatasetServiceError("No filename was provided.")

    if not file_utils.has_allowed_extension(filename):
        allowed = ", ".join(settings.ALLOWED_EXTENSIONS)
        raise DatasetServiceError(f"Unsupported file type. Allowed: {allowed}.")

    if not content:
        raise DatasetServiceError("The uploaded file is empty.")

    if len(content) > settings.MAX_UPLOAD_SIZE:
        max_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        raise DatasetServiceError(f"File exceeds the maximum size of {max_mb:.0f} MB.")

    # --- Persist file to disk ---------------------------------------------
    destination = file_utils.build_unique_path(filename)
    file_size = file_utils.save_bytes(content, destination)

    # --- Parse with pandas -------------------------------------------------
    try:
        df = _read_csv(destination)
    except DatasetServiceError:
        # Roll back the saved file so we do not leave orphaned uploads.
        destination.unlink(missing_ok=True)
        raise

    column_names = [str(col) for col in df.columns]
    preview = _build_preview(df)

    # --- Record metadata in SQLite ----------------------------------------
    dataset = Dataset(
        filename=file_utils.sanitize_filename(filename),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        file_size=file_size,
        dataset_path=str(destination),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset, column_names, preview


def read_dataset_csv(path: Path) -> pd.DataFrame:
    """Load a persisted dataset CSV from disk."""
    return _read_csv(path)


def list_datasets(db: Session) -> list[Dataset]:
    """Return all recorded datasets, newest first."""
    return db.query(Dataset).order_by(Dataset.upload_date.desc()).all()
