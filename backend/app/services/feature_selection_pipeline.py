"""Phase 6 — Feature Selection pipeline.

REQUIRES Phase 5 feature-engineered checkpoint. Never falls back to raw/cleaned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.schemas.feature_engineering import DatasetShapeSnapshot
from app.schemas.feature_selection import (
    FeatureSelectionAnalyzeRequest,
    FeatureSelectionApplyRequest,
    FeatureSelectionReport,
)
from app.services.feature_decision_engine import (
    build_feature_decisions,
    columns_to_drop_for_recommended_selection,
)
from app.services.feature_quality_analyzer import analyze_feature_quality
from app.services.feature_redundancy_analyzer import (
    analyze_correlation_pairs,
    categorical_exact_redundancy_notes,
)
from app.services.feature_selection_config import PREVIEW_ROWS
from app.services.feature_target_analyzer import (
    compute_mutual_information,
    detect_target_column,
    infer_target_task,
)
from app.services.feature_vif_analyzer import analyze_vif
from app.services.pipeline_state import (
    PipelineStateError,
    feature_engineered_dataset_path,
    feature_selected_dataset_path,
    feature_selection_report_path,
    load_feature_engineering_metadata,
    require_feature_engineered_dataframe,
    save_feature_selection_report,
)


class FeatureSelectionError(Exception):
    """Raised for expected Phase 6 failures."""


def _generated_meta_map(filename: str) -> dict[str, dict[str, Any]]:
    meta = load_feature_engineering_metadata(filename) or {}
    out: dict[str, dict[str, Any]] = {}
    for item in meta.get("generated_features") or []:
        name = item.get("feature")
        if name:
            out[str(name)] = item
    return out


def _build_preview(df: pd.DataFrame) -> list[dict[str, Any]]:
    head = df.head(PREVIEW_ROWS).where(pd.notnull(df.head(PREVIEW_ROWS)), None)
    return head.to_dict(orient="records")


def run_feature_selection_analysis(
    df: pd.DataFrame,
    *,
    dataset_id: int,
    filename: str,
    target_column: str | None = None,
) -> FeatureSelectionReport:
    """Analyze feature-engineered dataframe; do not write selected CSV."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise FeatureSelectionError("Feature-engineered dataset is empty or invalid.")

    generated = _generated_meta_map(filename)
    resolved_target, target_msg = detect_target_column(df, target_column)
    if target_column and resolved_target is None:
        raise FeatureSelectionError(target_msg or "Invalid target column.")

    quality_rows = analyze_feature_quality(
        df,
        target_column=resolved_target,
        generated_meta_by_name=generated,
    )

    target_scores = []
    mi_map: dict[str, float] = {}
    target_message = target_msg
    if resolved_target:
        target_scores, mi_err, mi_map = compute_mutual_information(
            df, resolved_target
        )
        if mi_err and not target_scores:
            target_message = mi_err
        task = infer_target_task(df[resolved_target])
    else:
        task = None

    correlation_pairs = analyze_correlation_pairs(
        df,
        quality_rows=quality_rows,
        target_column=resolved_target,
        mi_by_name=mi_map or None,
        generated_meta=generated,
    )
    cat_notes = categorical_exact_redundancy_notes(
        df, target_column=resolved_target
    )

    vif_rows, vif_available, vif_message = analyze_vif(
        df,
        quality_rows=quality_rows,
        target_column=resolved_target,
        correlation_pairs=correlation_pairs,
    )

    decisions, summary, explanations = build_feature_decisions(
        quality_rows=quality_rows,
        correlation_pairs=correlation_pairs,
        vif_rows=vif_rows,
        target_scores=target_scores,
        target_column=resolved_target,
    )
    summary.target_task = task
    summary.target_aware_applied = bool(target_scores)

    for note in cat_notes:
        if note not in explanations:
            explanations.append(note)

    before = DatasetShapeSnapshot(rows=len(df), columns=len(df.columns))
    return FeatureSelectionReport(
        dataset_id=dataset_id,
        source="feature_engineered",
        pipeline_stage_before="feature_engineered",
        summary=summary,
        quality_rows=quality_rows,
        correlation_pairs=correlation_pairs,
        vif_rows=vif_rows,
        vif_available=vif_available,
        vif_message=vif_message,
        target_scores=target_scores,
        target_message=target_message,
        decisions=decisions,
        explanations=explanations,
        before=before,
        after=None,
        applied=False,
        row_count_unchanged=True,
        feature_engineered_preserved=True,
        message=(
            "Phase 6 feature selection analysis complete on the "
            "feature-engineered dataset. No columns were removed yet."
        ),
    )


def analyze_feature_selection(
    db: Session,
    dataset_id: int,
    request: FeatureSelectionAnalyzeRequest | None = None,
) -> FeatureSelectionReport:
    """Load Phase 5 checkpoint and run analysis."""
    request = request or FeatureSelectionAnalyzeRequest()
    try:
        dataset, df, featured_path = require_feature_engineered_dataframe(
            db, dataset_id
        )
    except PipelineStateError as exc:
        raise FeatureSelectionError(str(exc)) from exc

    featured_bytes = featured_path.read_bytes()
    report = run_feature_selection_analysis(
        df,
        dataset_id=dataset_id,
        filename=dataset.filename,
        target_column=request.target_column,
    )
    if featured_path.read_bytes() != featured_bytes:
        raise FeatureSelectionError(
            "Safety check failed: feature-engineered checkpoint was modified."
        )
    return report


def apply_feature_selection(
    db: Session,
    dataset_id: int,
    request: FeatureSelectionApplyRequest | None = None,
) -> FeatureSelectionReport:
    """Analyze + apply recommended selection; write selected CSV + report."""
    request = request or FeatureSelectionApplyRequest()
    try:
        dataset, df, featured_path = require_feature_engineered_dataframe(
            db, dataset_id
        )
    except PipelineStateError as exc:
        raise FeatureSelectionError(str(exc)) from exc

    original_upload = Path(dataset.dataset_path)
    original_bytes = original_upload.read_bytes() if original_upload.exists() else None
    featured_bytes = featured_path.read_bytes()

    report = run_feature_selection_analysis(
        df,
        dataset_id=dataset_id,
        filename=dataset.filename,
        target_column=request.target_column,
    )

    if request.apply_recommended:
        drop_cols = columns_to_drop_for_recommended_selection(
            report.decisions,
            also_remove=request.also_remove,
            force_keep=request.force_keep,
            target_column=report.summary.target_column,
        )
    else:
        # Manual mode: only drop columns the user explicitly listed.
        protected = set(request.force_keep or [])
        if report.summary.target_column:
            protected.add(report.summary.target_column)
        drop_cols = [
            c
            for c in (request.also_remove or [])
            if c in df.columns and c not in protected
        ]

    # Ensure target never dropped; REVIEW only dropped via also_remove
    keep_cols = [c for c in df.columns if str(c) not in set(drop_cols)]
    if report.summary.target_column and report.summary.target_column not in keep_cols:
        keep_cols.append(report.summary.target_column)

    selected = df[keep_cols].copy()
    if len(selected) != len(df):
        raise FeatureSelectionError(
            f"Row count changed during feature selection ({len(df)} → {len(selected)})."
        )

    destination = feature_selected_dataset_path(dataset.filename)
    try:
        selected.to_csv(destination, index=False)
    except Exception as exc:
        raise FeatureSelectionError(
            f"Failed to save feature-selected dataset: {exc}"
        ) from exc

    after = DatasetShapeSnapshot(rows=len(selected), columns=len(selected.columns))
    report.after = after
    report.applied = True
    report.selected_filename = destination.name
    report.selected_path = str(destination)
    report.download_url = (
        f"/api/v1/datasets/{dataset_id}/feature-selection/download"
    )
    report.report_download_url = (
        f"/api/v1/datasets/{dataset_id}/feature-selection/report/download"
    )
    report.preview = _build_preview(selected)
    report.row_count_unchanged = after.rows == (report.before.rows if report.before else after.rows)
    report.message = (
        f"Feature selection applied. Removed {len(drop_cols)} recommended "
        f"feature(s). REVIEW features were kept unless explicitly removed. "
        "Feature-engineered checkpoint preserved."
    )

    report_path = save_feature_selection_report(
        dataset.filename,
        {
            **report.model_dump(),
            "phase": 6,
            "source_checkpoint": "feature_engineered",
            "feature_engineered_path": str(featured_path),
            "feature_selected_path": str(destination),
            "dropped_columns": drop_cols,
            "kept_columns": [str(c) for c in selected.columns],
        },
    )
    report.report_path = str(report_path)

    if original_bytes is not None and original_upload.read_bytes() != original_bytes:
        raise FeatureSelectionError(
            "Safety check failed: original upload was modified."
        )
    if featured_path.read_bytes() != featured_bytes:
        raise FeatureSelectionError(
            "Safety check failed: feature-engineered checkpoint was modified."
        )

    return report


def get_feature_selection_report(
    db: Session,
    dataset_id: int,
) -> FeatureSelectionReport:
    """Load saved Phase 6 report if present."""
    from app.services.pipeline_state import get_dataset_or_raise

    dataset = get_dataset_or_raise(db, dataset_id)
    path = feature_selection_report_path(dataset.filename)
    if not path.is_file():
        raise FeatureSelectionError(
            "No feature selection report found. Run Phase 6 analyze or apply first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Strip internal-only keys that are not part of the schema
    for key in (
        "phase",
        "source_checkpoint",
        "feature_engineered_path",
        "feature_selected_path",
        "dropped_columns",
        "kept_columns",
    ):
        payload.pop(key, None)
    return FeatureSelectionReport(**payload)


def get_selected_file_path(db: Session, dataset_id: int) -> Path:
    from app.services.pipeline_state import get_dataset_or_raise

    dataset = get_dataset_or_raise(db, dataset_id)
    path = feature_selected_dataset_path(dataset.filename)
    if not path.is_file():
        raise FeatureSelectionError(
            "No feature-selected dataset found. Apply Phase 6 feature selection first."
        )
    return path


def get_selection_report_file_path(db: Session, dataset_id: int) -> Path:
    from app.services.pipeline_state import get_dataset_or_raise

    dataset = get_dataset_or_raise(db, dataset_id)
    path = feature_selection_report_path(dataset.filename)
    if not path.is_file():
        raise FeatureSelectionError(
            "No feature selection report file found."
        )
    return path
