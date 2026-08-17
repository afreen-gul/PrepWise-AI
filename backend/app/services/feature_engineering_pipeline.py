"""Phase 5 full pipeline: datetime → numerical → text → validation.

REQUIRES Phase 4 cleaned dataset checkpoint. Never falls back to raw upload.

Feature engineering is optional: recommend candidates → user selects → generate
only selected features (or pass-through cleaned columns when none selected).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.schemas.feature_engineering import (
    DatasetShapeSnapshot,
    DatetimeIssueFlag,
    FeatureCandidate,
    FeatureEngineeringApplyRequest,
    FeatureEngineeringCandidateReport,
    Phase5FeatureEngineeringResult,
)
from app.services.datetime_feature_engineer import engineer_datetime_features
from app.services.feature_engineering_config import PREVIEW_ROWS
from app.services.feature_validator import (
    FeatureValidationError,
    validate_and_finalize_features,
)
from app.services.numerical_feature_engineer import engineer_numerical_features
from app.services.pipeline_state import (
    PipelineStateError,
    feature_engineered_dataset_path,
    require_cleaned_dataframe,
    save_feature_engineering_metadata,
)
from app.services.text_feature_engineer import engineer_text_features


class FeatureEngineeringPipelineError(Exception):
    """Raised for expected Phase 5 pipeline failures."""


def _priority_for_generated(category: str, transformation: str) -> tuple[str, bool]:
    """Return (priority, default_selected)."""
    t = (transformation or "").lower()
    if category == "numerical" and ("log" in t or "/" in t or "ratio" in t):
        return "HIGH", True
    if category == "datetime":
        return "MEDIUM", True
    if category == "text":
        return "MEDIUM", True
    if "bin" in t:
        return "MEDIUM", True
    return "MEDIUM", True


def run_feature_engineering_pipeline(
    df: pd.DataFrame,
    *,
    selected_feature_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Phase 5.2–5.5 on a copy of the *cleaned* dataframe.

    selected_feature_ids:
      - None → generate all viable features (legacy / discovery)
      - [] → return cleaned copy unchanged (pass-through)
      - [...] → generate only those feature names
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise FeatureEngineeringPipelineError("Dataset could not be loaded.")

    working = df.copy()
    original_columns = [str(c) for c in working.columns]
    expected_rows = len(working)

    if selected_feature_ids is not None and len(selected_feature_ids) == 0:
        return working, {
            "original_columns": original_columns,
            "generated": [],
            "skipped": [],
            "removed": [],
            "issues": [],
            "datetime_analyses": [],
            "polynomial_recommendations": [],
            "datetime_features_generated": 0,
            "numerical_features_generated": 0,
            "text_features_generated": 0,
            "expected_rows": expected_rows,
            "selection_mode": "pass_through",
        }

    selected_set: set[str] | None = (
        None if selected_feature_ids is None else set(selected_feature_ids)
    )

    all_generated = []
    all_skipped = []
    all_issues: list[DatetimeIssueFlag] = []
    datetime_analyses = []
    polynomial_recommendations: list[str] = []

    # 5.2 Datetime
    working, dt_details = engineer_datetime_features(working, selected=selected_set)
    for g in dt_details["generated"]:
        g.category = g.category or "datetime"
        g.phase = 5
        all_generated.append(g)
    all_skipped.extend(dt_details["skipped"])
    all_issues.extend(dt_details["issues"])
    datetime_analyses = dt_details["analyses"]

    # 5.3 Numerical
    working, num_details = engineer_numerical_features(working, selected=selected_set)
    for g in num_details["generated"]:
        g.phase = 5
        all_generated.append(g)
    all_skipped.extend(num_details["skipped"])
    for issue in num_details["issues"]:
        all_issues.append(
            DatetimeIssueFlag(
                issue=str(issue.get("issue", "numerical_issue")),
                columns=list(issue.get("columns") or []),
                count=int(issue.get("count", 0)),
                message=str(issue.get("message", "")),
            )
        )
    polynomial_recommendations.extend(num_details.get("polynomial_recommendations") or [])

    # 5.4 Text
    working, text_details = engineer_text_features(working, selected=selected_set)
    for g in text_details["generated"]:
        g.phase = 5
        all_generated.append(g)
    all_skipped.extend(text_details["skipped"])

    if len(working) != expected_rows:
        raise FeatureEngineeringPipelineError(
            f"Row count changed during feature engineering "
            f"({expected_rows} → {len(working)})."
        )

    # 5.5 Validation
    try:
        working, val_details = validate_and_finalize_features(
            working,
            original_columns=original_columns,
            generated_meta=all_generated,
            expected_rows=expected_rows,
        )
    except FeatureValidationError as exc:
        raise FeatureEngineeringPipelineError(str(exc)) from exc

    all_issues.extend(val_details["issues"])
    kept = val_details["kept_generated"]
    removed = val_details["removed"]

    dt_count = sum(1 for g in kept if g.category == "datetime")
    num_count = sum(1 for g in kept if g.category == "numerical")
    text_count = sum(1 for g in kept if g.category == "text")

    mode = "all" if selected_feature_ids is None else "selected"
    return working, {
        "original_columns": original_columns,
        "generated": kept,
        "skipped": all_skipped,
        "removed": removed,
        "issues": all_issues,
        "datetime_analyses": datetime_analyses,
        "polynomial_recommendations": polynomial_recommendations,
        "datetime_features_generated": dt_count,
        "numerical_features_generated": num_count,
        "text_features_generated": text_count,
        "expected_rows": expected_rows,
        "selection_mode": mode,
    }


def discover_feature_candidates(
    db: Session,
    dataset_id: int,
) -> FeatureEngineeringCandidateReport:
    """Dry-run Phase 5 on cleaned data to list concrete selectable candidates."""
    try:
        dataset, df, _ = require_cleaned_dataframe(db, dataset_id)
    except PipelineStateError as exc:
        raise FeatureEngineeringPipelineError(str(exc)) from exc

    _, details = run_feature_engineering_pipeline(df, selected_feature_ids=None)

    candidates: list[FeatureCandidate] = []
    for g in details["generated"]:
        priority, default_selected = _priority_for_generated(
            g.category or "", g.transformation or ""
        )
        candidates.append(
            FeatureCandidate(
                id=g.feature,
                feature=g.feature,
                source=g.source,
                category=g.category or "unknown",
                transformation=g.transformation,
                reason=g.reason,
                priority=priority,
                feature_type=g.feature_type or "Integer",
                default_selected=default_selected,
            )
        )

    from app.services.dataset_profiler import _name_matches_target

    potential_targets = [
        str(c) for c in df.columns if _name_matches_target(str(c))
    ]

    recommended = len(candidates) > 0
    if recommended:
        message = (
            f"Found {len(candidates)} candidate engineered feature(s). "
            "Select which ones to create — feature engineering is optional."
        )
    else:
        message = (
            "No additional feature engineering is recommended for this cleaned "
            "dataset. You can continue without creating new features "
            "(cleaned columns will be passed through to Phase 6)."
        )

    return FeatureEngineeringCandidateReport(
        dataset_id=dataset_id,
        source="cleaned",
        columns_analyzed=len(df.columns),
        candidates=candidates,
        candidates_count=len(candidates),
        engineering_recommended=recommended,
        potential_targets=potential_targets,
        polynomial_recommendations=details.get("polynomial_recommendations") or [],
        transformations_applied=False,
        message=message,
    )


def _build_preview(df: pd.DataFrame) -> list[dict[str, Any]]:
    head = df.head(PREVIEW_ROWS).where(pd.notnull(df.head(PREVIEW_ROWS)), None)
    return head.to_dict(orient="records")


def apply_phase5_feature_engineering(
    db: Session,
    dataset_id: int,
    request: FeatureEngineeringApplyRequest | None = None,
) -> Phase5FeatureEngineeringResult:
    """Run Phase 5 on the Phase-4 cleaned checkpoint only.

    When selected_feature_ids is [] the cleaned frame is copied to featured
    without new columns so Phase 6 can proceed.
    When selected_feature_ids is None (default), all viable features are created
    (backward-compatible with existing tests).
    """
    request = request or FeatureEngineeringApplyRequest()
    try:
        dataset, df, cleaned_path = require_cleaned_dataframe(db, dataset_id)
    except PipelineStateError as exc:
        raise FeatureEngineeringPipelineError(str(exc)) from exc

    original_upload = Path(dataset.dataset_path)
    original_bytes = original_upload.read_bytes() if original_upload.exists() else None
    cleaned_bytes = cleaned_path.read_bytes()

    before = DatasetShapeSnapshot(rows=len(df), columns=len(df.columns))
    engineered, details = run_feature_engineering_pipeline(
        df,
        selected_feature_ids=request.selected_feature_ids,
    )
    after = DatasetShapeSnapshot(rows=len(engineered), columns=len(engineered.columns))

    destination = feature_engineered_dataset_path(dataset.filename)
    try:
        engineered.to_csv(destination, index=False)
    except Exception as exc:
        raise FeatureEngineeringPipelineError(
            f"Failed to save feature-engineered dataset: {exc}"
        ) from exc

    selection_mode = details.get("selection_mode", "all")
    metadata = {
        "dataset_id": dataset_id,
        "phase": 5,
        "source_checkpoint": "cleaned",
        "cleaned_path": str(cleaned_path),
        "feature_engineered_path": str(destination),
        "before": before.model_dump(),
        "after": after.model_dump(),
        "selected_feature_ids": request.selected_feature_ids,
        "selection_mode": selection_mode,
        "generated_features": [g.model_dump() for g in details["generated"]],
        "skipped_features": [s.model_dump() for s in details["skipped"]],
        "removed_features": [r.model_dump() for r in details["removed"]],
        "input_columns": details["original_columns"],
        "output_columns": [str(c) for c in engineered.columns],
    }
    meta_path = save_feature_engineering_metadata(dataset.filename, metadata)

    if original_bytes is not None and original_upload.read_bytes() != original_bytes:
        raise FeatureEngineeringPipelineError(
            "Safety check failed: original upload was modified."
        )
    if cleaned_path.read_bytes() != cleaned_bytes:
        raise FeatureEngineeringPipelineError(
            "Safety check failed: cleaned checkpoint was modified."
        )

    if selection_mode == "pass_through":
        message = (
            "No engineered features were created. Cleaned columns were passed "
            "through to the feature-engineered checkpoint for Phase 6. "
            "Original upload and cleaned checkpoint were not modified."
        )
    elif selection_mode == "selected":
        message = (
            f"Created {len(details['generated'])} user-selected engineered "
            "feature(s) on the cleaned dataset. Original upload and cleaned "
            "checkpoint were not modified."
        )
    else:
        message = (
            "Feature engineering applied to the cleaned dataset. "
            "Original upload and cleaned checkpoint were not modified."
        )

    return Phase5FeatureEngineeringResult(
        dataset_id=dataset_id,
        source="cleaned",
        pipeline_stage_before="cleaned",
        pipeline_stage_after="feature_engineered",
        applied_to_cleaned_dataset=True,
        selected_feature_ids=request.selected_feature_ids,
        selection_mode=selection_mode,
        original_feature_count=before.columns,
        datetime_features_generated=details["datetime_features_generated"],
        numerical_features_generated=details["numerical_features_generated"],
        text_features_generated=details["text_features_generated"],
        features_generated=len(details["generated"]),
        features_skipped=len(details["skipped"]),
        features_removed=len(details["removed"]),
        final_feature_count=after.columns,
        before=before,
        after=after,
        generated_features=details["generated"],
        skipped_features=details["skipped"],
        removed_features=details["removed"],
        issues=details["issues"],
        datetime_analyses=details["datetime_analyses"],
        polynomial_recommendations=details["polynomial_recommendations"],
        original_columns_preserved=True,
        row_count_unchanged=before.rows == after.rows,
        featured_filename=destination.name,
        featured_path=str(destination),
        metadata_path=str(meta_path),
        preview=_build_preview(engineered),
        download_url=f"/api/v1/datasets/{dataset_id}/feature-engineering/download",
        message=message,
    )


def get_featured_pipeline_file_path(db: Session, dataset_id: int) -> Path:
    from app.services.pipeline_state import get_dataset_or_raise

    dataset = get_dataset_or_raise(db, dataset_id)
    path = feature_engineered_dataset_path(dataset.filename)
    if not path.is_file():
        raise FeatureEngineeringPipelineError(
            "No feature-engineered dataset found. Apply Phase 5 feature engineering first."
        )
    return path

