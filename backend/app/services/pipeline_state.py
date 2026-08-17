"""PrepWise cumulative pipeline state (Phase 1–6 checkpoints).



Raw upload is immutable. Later phases consume prior checkpoints only.

"""



from __future__ import annotations



import json

import re

from enum import Enum

from pathlib import Path

from typing import Any



import pandas as pd

from sqlalchemy.orm import Session



from app.core.config import settings

from app.models.dataset import Dataset

from app.services.data_cleaner import DataCleanerError, get_cleaned_file_path

from app.services.dataset_service import DatasetServiceError, read_dataset_csv





class PipelineStage(str, Enum):

    RAW = "raw"

    CLEANED = "cleaned"

    FEATURE_ENGINEERED = "feature_engineered"

    FEATURE_SELECTED = "feature_selected"





class PipelineStateError(Exception):

    """Raised when a required pipeline checkpoint is missing or invalid."""





def _safe_stem(original_filename: str) -> tuple[str, str]:

    stem = Path(original_filename).stem

    suffix = Path(original_filename).suffix or ".csv"

    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem) or "dataset"

    return safe_stem, suffix





def cleaned_dataset_path(original_filename: str) -> Path:

    safe_stem, suffix = _safe_stem(original_filename)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    return settings.processed_dir / f"cleaned_{safe_stem}{suffix}"





def feature_engineered_dataset_path(original_filename: str) -> Path:

    """Feature-engineered export (existing ``featured_`` naming convention)."""

    safe_stem, suffix = _safe_stem(original_filename)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    return settings.processed_dir / f"featured_{safe_stem}{suffix}"





def feature_metadata_path(original_filename: str) -> Path:

    safe_stem, _ = _safe_stem(original_filename)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    return settings.processed_dir / f"featured_{safe_stem}_metadata.json"





def feature_selected_dataset_path(original_filename: str) -> Path:

    safe_stem, suffix = _safe_stem(original_filename)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    return settings.processed_dir / f"selected_{safe_stem}{suffix}"





def feature_selection_report_path(original_filename: str) -> Path:

    safe_stem, _ = _safe_stem(original_filename)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    return settings.processed_dir / f"selected_{safe_stem}_report.json"





def get_dataset_or_raise(db: Session, dataset_id: int) -> Dataset:

    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if dataset is None:

        raise PipelineStateError(f"Dataset with id {dataset_id} was not found.")

    return dataset





def raw_dataset_path(dataset: Dataset) -> Path:

    return Path(dataset.dataset_path)





def has_cleaned_dataset(db: Session, dataset_id: int) -> bool:

    try:

        path = get_cleaned_file_path(db, dataset_id)

    except DataCleanerError:

        return False

    return path.is_file()





def has_feature_engineered_dataset(db: Session, dataset_id: int) -> bool:

    dataset = get_dataset_or_raise(db, dataset_id)

    return feature_engineered_dataset_path(dataset.filename).is_file()





def has_feature_selected_dataset(db: Session, dataset_id: int) -> bool:

    dataset = get_dataset_or_raise(db, dataset_id)

    return feature_selected_dataset_path(dataset.filename).is_file()





def load_raw_dataframe(db: Session, dataset_id: int) -> pd.DataFrame:

    """Load the immutable raw upload (never modified by later phases)."""

    dataset = get_dataset_or_raise(db, dataset_id)

    path = raw_dataset_path(dataset)

    if not path.is_file():

        raise PipelineStateError("Original uploaded dataset file is missing.")

    try:

        return read_dataset_csv(path)

    except DatasetServiceError as exc:

        raise PipelineStateError(str(exc)) from exc





def require_cleaned_dataframe(

    db: Session,

    dataset_id: int,

) -> tuple[Dataset, pd.DataFrame, Path]:

    """Load Phase 4 cleaned checkpoint. Never falls back to raw data."""

    dataset = get_dataset_or_raise(db, dataset_id)

    try:

        path = get_cleaned_file_path(db, dataset_id)

    except DataCleanerError as exc:

        raise PipelineStateError(

            "Complete Phase 4 cleaning before running Feature Engineering. "

            "No cleaned dataset checkpoint was found."

        ) from exc



    if not path.is_file():

        raise PipelineStateError(

            "Complete Phase 4 cleaning before running Feature Engineering. "

            "No cleaned dataset checkpoint was found."

        )



    before_bytes = path.read_bytes()

    df = pd.read_csv(path)

    if path.read_bytes() != before_bytes:

        raise PipelineStateError(

            "Cleaned dataset was modified unexpectedly while loading."

        )

    return dataset, df, path





def require_feature_engineered_dataframe(

    db: Session,

    dataset_id: int,

) -> tuple[Dataset, pd.DataFrame, Path]:

    """Load Phase 5 featured checkpoint for Phase 6. Never falls back to raw."""

    dataset = get_dataset_or_raise(db, dataset_id)

    path = feature_engineered_dataset_path(dataset.filename)

    if not path.is_file():

        raise PipelineStateError(

            "Complete Phase 5 feature engineering before running Feature Selection. "

            "No feature-engineered dataset checkpoint was found."

        )

    before_bytes = path.read_bytes()

    df = pd.read_csv(path)

    if path.read_bytes() != before_bytes:

        raise PipelineStateError(

            "Feature-engineered dataset was modified unexpectedly while loading."

        )

    return dataset, df, path





def require_feature_selected_dataframe(

    db: Session,

    dataset_id: int,

) -> tuple[Dataset, pd.DataFrame, Path]:

    """Load Phase 6 selected checkpoint (for future Phase 7 hand-off)."""

    dataset = get_dataset_or_raise(db, dataset_id)

    path = feature_selected_dataset_path(dataset.filename)

    if not path.is_file():

        raise PipelineStateError(

            "Complete Phase 6 feature selection before continuing. "

            "No feature-selected dataset checkpoint was found."

        )

    before_bytes = path.read_bytes()

    df = pd.read_csv(path)

    if path.read_bytes() != before_bytes:

        raise PipelineStateError(

            "Feature-selected dataset was modified unexpectedly while loading."

        )

    return dataset, df, path





def current_pipeline_stage(db: Session, dataset_id: int) -> PipelineStage:

    if has_feature_selected_dataset(db, dataset_id):

        return PipelineStage.FEATURE_SELECTED

    if has_feature_engineered_dataset(db, dataset_id):

        return PipelineStage.FEATURE_ENGINEERED

    if has_cleaned_dataset(db, dataset_id):

        return PipelineStage.CLEANED

    return PipelineStage.RAW





def save_feature_engineering_metadata(

    original_filename: str,

    metadata: dict[str, Any],

) -> Path:

    path = feature_metadata_path(original_filename)

    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return path





def load_feature_engineering_metadata(

    original_filename: str,

) -> dict[str, Any] | None:

    path = feature_metadata_path(original_filename)

    if not path.is_file():

        return None

    return json.loads(path.read_text(encoding="utf-8"))





def save_feature_selection_report(

    original_filename: str,

    report: dict[str, Any],

) -> Path:

    path = feature_selection_report_path(original_filename)

    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return path





def load_feature_selection_report(

    original_filename: str,

) -> dict[str, Any] | None:

    path = feature_selection_report_path(original_filename)

    if not path.is_file():

        return None

    return json.loads(path.read_text(encoding="utf-8"))





def build_pipeline_status(db: Session, dataset_id: int) -> dict[str, Any]:

    """User-facing pipeline status for API/UI."""

    dataset = get_dataset_or_raise(db, dataset_id)

    raw_path = raw_dataset_path(dataset)

    cleaned_exists = has_cleaned_dataset(db, dataset_id)

    featured_exists = has_feature_engineered_dataset(db, dataset_id)

    selected_exists = has_feature_selected_dataset(db, dataset_id)

    stage = current_pipeline_stage(db, dataset_id)



    cleaned_path = cleaned_dataset_path(dataset.filename)

    featured_path = feature_engineered_dataset_path(dataset.filename)

    meta_path = feature_metadata_path(dataset.filename)

    selected_path = feature_selected_dataset_path(dataset.filename)

    selection_report = feature_selection_report_path(dataset.filename)



    stage_label = {

        PipelineStage.RAW: "Raw Dataset",

        PipelineStage.CLEANED: "Cleaned Dataset",

        PipelineStage.FEATURE_ENGINEERED: "Feature-Engineered Dataset",

        PipelineStage.FEATURE_SELECTED: "Feature-Selected Dataset",

    }[stage]



    phases = [

        {

            "phase": 1,

            "name": "Upload / Dataset Understanding",

            "status": "complete" if raw_path.is_file() else "not_started",

        },

        {

            "phase": 2,

            "name": "Dataset Profiling",

            "status": "complete" if raw_path.is_file() else "not_started",

            "note": "Profiling is available once a dataset is uploaded.",

        },

        {

            "phase": 3,

            "name": "Data Quality Analysis",

            "status": "complete" if raw_path.is_file() else "not_started",

            "note": "Quality assessment is available once a dataset is uploaded.",

        },

        {

            "phase": 4,

            "name": "Cleaning + Preprocessing",

            "status": "complete" if cleaned_exists else "not_started",

            "output": "cleaned_dataset" if cleaned_exists else None,

            "path": str(cleaned_path) if cleaned_exists else None,

        },

        {

            "phase": 5,

            "name": "Feature Engineering",

            "status": "complete" if featured_exists else "not_started",

            "requires": "Phase 4 cleaned dataset",

            "output": "feature_engineered_dataset" if featured_exists else None,

            "path": str(featured_path) if featured_exists else None,

            "metadata_path": str(meta_path) if meta_path.is_file() else None,

        },

        {

            "phase": 6,

            "name": "Feature Selection",

            "status": "complete" if selected_exists else (

                "ready" if featured_exists else "not_started"

            ),

            "requires": "Phase 5 feature-engineered dataset",

            "output": "feature_selected_dataset" if selected_exists else None,

            "path": str(selected_path) if selected_exists else None,

            "metadata_path": (

                str(selection_report) if selection_report.is_file() else None

            ),

        },

    ]



    return {

        "dataset_id": dataset_id,

        "filename": dataset.filename,

        "current_stage": stage.value,

        "current_stage_label": stage_label,

        "raw_immutable": True,

        "raw_path": str(raw_path),

        "cleaned_available": cleaned_exists,

        "feature_engineered_available": featured_exists,

        "feature_selected_available": selected_exists,

        "phase5_ready": cleaned_exists,

        "phase6_ready": featured_exists,

        "phase7_ready": selected_exists,

        "exports": {

            "raw": str(raw_path) if raw_path.is_file() else None,

            "cleaned": str(cleaned_path) if cleaned_exists else None,

            "feature_engineered": str(featured_path) if featured_exists else None,

            "feature_selected": str(selected_path) if selected_exists else None,

            "feature_selection_report": (

                str(selection_report) if selection_report.is_file() else None

            ),

        },

        "phases": phases,

        "message": (

            f"Current pipeline dataset: {stage_label}. "

            "Raw upload remains immutable."

        ),

    }


