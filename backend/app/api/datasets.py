"""API routes for dataset upload and retrieval.

Routes are intentionally thin: they handle HTTP concerns only and delegate all
business logic to ``services.dataset_service``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.cleaning import (
    CleaningApplyResponse,
    CleaningConfig,
    CleaningPreviewResponse,
)
from app.schemas.dataset import DatasetRead, DatasetUploadResponse
from app.schemas.feature_engineering import (
    DatetimeFeatureEngineeringResult,
    FeatureEngineeringApplyRequest,
    FeatureEngineeringCandidateReport,
    FeatureEngineeringOpportunityReport,
    Phase5FeatureEngineeringResult,
    PipelineStatusResponse,
)
from app.schemas.feature_selection import (
    FeatureSelectionAnalyzeRequest,
    FeatureSelectionApplyRequest,
    FeatureSelectionReport,
)
from app.schemas.profile import DatasetProfile
from app.schemas.quality import DataQualityReport
from app.services import dataset_service
from app.services.data_cleaner import (
    DataCleanerError,
    apply_cleaning,
    get_cleaned_file_path,
    preview_cleaning,
)
from app.services.data_quality import DataQualityError, build_quality_report
from app.services.dataset_profiler import DatasetProfilerError, build_profile
from app.services.dataset_service import DatasetServiceError
from app.services.feature_opportunity_detector import (
    FeatureOpportunityError,
    build_feature_opportunity_report,
)
from app.services.datetime_feature_engineer import (
    DatetimeFeatureEngineeringError,
    apply_datetime_feature_engineering,
    get_featured_file_path,
)
from app.services.feature_engineering_pipeline import (
    FeatureEngineeringPipelineError,
    apply_phase5_feature_engineering,
    discover_feature_candidates,
    get_featured_pipeline_file_path,
)
from app.services.feature_selection_pipeline import (
    FeatureSelectionError,
    analyze_feature_selection,
    apply_feature_selection,
    get_feature_selection_report,
    get_selected_file_path,
    get_selection_report_file_path,
)
from app.services.pipeline_state import PipelineStateError, build_pipeline_status

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CSV dataset",
)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DatasetUploadResponse:
    """Accept a CSV upload, store it, parse it, and return an overview."""
    content = await file.read()

    try:
        dataset, column_names, preview = dataset_service.process_upload(
            db,
            filename=file.filename or "",
            content=content,
        )
    except DatasetServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return DatasetUploadResponse(
        dataset=DatasetRead.model_validate(dataset),
        column_names=column_names,
        preview=preview,
    )


@router.get(
    "",
    response_model=list[DatasetRead],
    summary="List uploaded datasets",
)
def list_datasets(db: Session = Depends(get_db)) -> list[DatasetRead]:
    """Return metadata for all previously uploaded datasets."""
    datasets = dataset_service.list_datasets(db)
    return [DatasetRead.model_validate(d) for d in datasets]


@router.post(
    "/{dataset_id}/profile",
    response_model=DatasetProfile,
    summary="Generate dataset profile",
)
def generate_dataset_profile(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> DatasetProfile:
    """Analyze the dataset without modifying it."""
    try:
        return build_profile(db, dataset_id)
    except DatasetProfilerError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/quality",
    response_model=DataQualityReport,
    summary="Assess data quality",
)
def assess_dataset_quality(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> DataQualityReport:
    """Detect and report data quality issues without modifying the dataset."""
    try:
        return build_quality_report(db, dataset_id)
    except DataQualityError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/clean/preview",
    response_model=CleaningPreviewResponse,
    summary="Preview data cleaning (dry run)",
)
def preview_dataset_cleaning(
    dataset_id: int,
    config: CleaningConfig,
    db: Session = Depends(get_db),
) -> CleaningPreviewResponse:
    """Simulate cleaning without writing files or modifying the original."""
    try:
        return preview_cleaning(db, dataset_id, config)
    except DataCleanerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/clean",
    response_model=CleaningApplyResponse,
    summary="Apply data cleaning",
)
def apply_dataset_cleaning(
    dataset_id: int,
    config: CleaningConfig,
    db: Session = Depends(get_db),
) -> CleaningApplyResponse:
    """Clean a copy of the dataset and save it under ``processed/``."""
    try:
        return apply_cleaning(db, dataset_id, config)
    except DataCleanerError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/{dataset_id}/clean/download",
    summary="Download cleaned dataset",
)
def download_cleaned_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Serve the cleaned CSV copy (original remains untouched)."""
    try:
        path = get_cleaned_file_path(db, dataset_id)
    except DataCleanerError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return FileResponse(
        path=path,
        media_type="text/csv",
        filename=path.name,
    )


@router.post(
    "/{dataset_id}/feature-opportunities",
    response_model=FeatureEngineeringOpportunityReport,
    summary="Detect feature types and engineering opportunities (Phase 5.1)",
)
def detect_feature_opportunities(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> FeatureEngineeringOpportunityReport:
    """Analyze cleaned Phase-4 checkpoint — recommendations only."""
    try:
        return build_feature_opportunity_report(db, dataset_id)
    except FeatureOpportunityError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/{dataset_id}/pipeline-status",
    response_model=PipelineStatusResponse,
    summary="Get cumulative pipeline checkpoint status",
)
def get_pipeline_status(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> PipelineStatusResponse:
    """Report raw / cleaned / feature-engineered availability for this dataset."""
    try:
        payload = build_pipeline_status(db, dataset_id)
    except PipelineStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return PipelineStatusResponse(**payload)

@router.post(
    "/{dataset_id}/feature-engineering/candidates",
    response_model=FeatureEngineeringCandidateReport,
    summary="Recommend concrete engineered feature candidates (Phase 5)",
)
def recommend_feature_candidates(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> FeatureEngineeringCandidateReport:
    """Dry-run on cleaned data — recommendations only, nothing is written."""
    try:
        return discover_feature_candidates(db, dataset_id)
    except FeatureEngineeringPipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/feature-engineering/datetime",
    response_model=DatetimeFeatureEngineeringResult,
    summary="Apply datetime feature engineering (Phase 5.2)",
)
def apply_datetime_features(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> DatetimeFeatureEngineeringResult:
    """Create datetime-derived features on a working copy; save featured CSV."""
    try:
        return apply_datetime_feature_engineering(db, dataset_id)
    except DatetimeFeatureEngineeringError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/feature-engineering",
    response_model=Phase5FeatureEngineeringResult,
    summary="Generate selected (or all) Phase 5 engineered features",
)
def apply_full_feature_engineering(
    dataset_id: int,
    request: FeatureEngineeringApplyRequest = FeatureEngineeringApplyRequest(),
    db: Session = Depends(get_db),
) -> Phase5FeatureEngineeringResult:
    """Generate user-selected features on cleaned data; empty list = pass-through."""
    try:
        return apply_phase5_feature_engineering(db, dataset_id, request)
    except FeatureEngineeringPipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/{dataset_id}/feature-engineering/datetime/download",
    summary="Download featured dataset (Phase 5.2)",
)
def download_featured_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Serve the featured CSV (original and cleaned files remain untouched)."""
    try:
        path = get_featured_file_path(db, dataset_id)
    except DatetimeFeatureEngineeringError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return FileResponse(
        path=path,
        media_type="text/csv",
        filename=path.name,
    )


@router.get(
    "/{dataset_id}/feature-engineering/download",
    summary="Download featured dataset (full Phase 5)",
)
def download_phase5_featured_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Serve the Phase-5 featured CSV."""
    try:
        path = get_featured_pipeline_file_path(db, dataset_id)
    except FeatureEngineeringPipelineError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return FileResponse(
        path=path,
        media_type="text/csv",
        filename=path.name,
    )


@router.post(
    "/{dataset_id}/feature-selection/analyze",
    response_model=FeatureSelectionReport,
    summary="Analyze features for selection (Phase 6)",
)
def analyze_dataset_feature_selection(
    dataset_id: int,
    request: FeatureSelectionAnalyzeRequest = FeatureSelectionAnalyzeRequest(),
    db: Session = Depends(get_db),
) -> FeatureSelectionReport:
    """Run Phase 6 analysis on the feature-engineered checkpoint (no column drops)."""
    try:
        return analyze_feature_selection(db, dataset_id, request)
    except FeatureSelectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/feature-selection/apply",
    response_model=FeatureSelectionReport,
    summary="Apply recommended feature selection (Phase 6)",
)
def apply_dataset_feature_selection(
    dataset_id: int,
    request: FeatureSelectionApplyRequest = FeatureSelectionApplyRequest(),
    db: Session = Depends(get_db),
) -> FeatureSelectionReport:
    """Drop REMOVE recommendations; keep REVIEW unless explicitly removed."""
    try:
        return apply_feature_selection(db, dataset_id, request)
    except FeatureSelectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "/{dataset_id}/feature-selection/report",
    response_model=FeatureSelectionReport,
    summary="Get saved feature selection report",
)
def get_dataset_feature_selection_report(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> FeatureSelectionReport:
    try:
        return get_feature_selection_report(db, dataset_id)
    except FeatureSelectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/{dataset_id}/feature-selection/download",
    summary="Download feature-selected dataset",
)
def download_feature_selected_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> FileResponse:
    try:
        path = get_selected_file_path(db, dataset_id)
    except FeatureSelectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return FileResponse(
        path=path,
        media_type="text/csv",
        filename=path.name,
    )


@router.get(
    "/{dataset_id}/feature-selection/report/download",
    summary="Download feature selection report JSON",
)
def download_feature_selection_report_file(
    dataset_id: int,
    db: Session = Depends(get_db),
) -> FileResponse:
    try:
        path = get_selection_report_file_path(db, dataset_id)
    except FeatureSelectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return FileResponse(
        path=path,
        media_type="application/json",
        filename=path.name,
    )
