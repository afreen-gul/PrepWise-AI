"""Single source of truth for PrepWise pipeline navigation and stage state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

# Ordered workflow stages (excludes "home")
STAGE_ORDER: tuple[str, ...] = (
    "upload",
    "profile",
    "quality",
    "cleaning",
    "engineer",
    "select",
    "export",
)

STAGE_LABELS: dict[str, str] = {
    "upload": "Upload",
    "profile": "Profile",
    "quality": "Data Quality",
    "cleaning": "Cleaning",
    "engineer": "Feature Engineering",
    "select": "Feature Selection",
    "export": "Export",
}

# Session keys cleared when dataset is replaced/removed
DATASET_SESSION_KEYS: tuple[str, ...] = (
    "upload_payload",
    "upload_cache_key",
    "uploaded_file_name",
    "uploaded_file_bytes",
    "profile_payload",
    "profile_dataset_id",
    "quality_payload",
    "quality_dataset_id",
    "clean_preview",
    "clean_preview_dataset_id",
    "clean_result",
    "clean_result_dataset_id",
    "feature_opp_payload",
    "feature_opp_dataset_id",
    "fe_candidates_payload",
    "fe_candidates_dataset_id",
    "datetime_fe_payload",
    "datetime_fe_dataset_id",
    "phase5_fe_payload",
    "phase5_fe_dataset_id",
    "fs_analyze_payload",
    "fs_analyze_dataset_id",
    "fs_apply_payload",
    "fs_apply_dataset_id",
    "pipeline_status",
    "pipeline_status_dataset_id",
)


@dataclass(frozen=True)
class PipelineContext:
    """Derived pipeline state for the active dataset."""

    has_dataset: bool
    dataset_id: int | None
    filename: str | None
    pipeline: dict[str, Any] | None
    active_nav: str
    completed: dict[str, bool]
    accessible: dict[str, bool]
    current_stage: str
    profile_done: bool
    quality_done: bool
    cleaning_done: bool
    engineer_done: bool
    select_done: bool


def _dataset_id() -> int | None:
    payload = st.session_state.get("upload_payload")
    if not payload:
        return None
    dataset = payload.get("dataset") or {}
    return dataset.get("id")


def _session_matches_dataset(key: str, dataset_id: int | None) -> bool:
    if dataset_id is None:
        return False
    return st.session_state.get(key) == dataset_id


def build_pipeline_context(
    *,
    active_nav: str,
    pipeline: dict[str, Any] | None,
) -> PipelineContext:
    """Derive completion, accessibility, and current stage from session + backend."""
    has_dataset = "upload_payload" in st.session_state
    dataset_id = _dataset_id()
    pipeline = pipeline or {}

    profile_done = _session_matches_dataset("profile_dataset_id", dataset_id) and bool(
        st.session_state.get("profile_payload")
    )
    quality_done = _session_matches_dataset("quality_dataset_id", dataset_id) and bool(
        st.session_state.get("quality_payload")
    )
    cleaning_done = bool(pipeline.get("cleaned_available")) or (
        _session_matches_dataset("clean_result_dataset_id", dataset_id)
        and bool(st.session_state.get("clean_result"))
    )
    engineer_done = bool(pipeline.get("feature_engineered_available")) or (
        _session_matches_dataset("phase5_fe_dataset_id", dataset_id)
        and bool(st.session_state.get("phase5_fe_payload"))
    )
    select_done = bool(pipeline.get("feature_selected_available")) or (
        _session_matches_dataset("fs_apply_dataset_id", dataset_id)
        and bool(st.session_state.get("fs_apply_payload"))
    )

    completed = {
        "upload": has_dataset,
        "profile": profile_done,
        "quality": quality_done,
        "cleaning": cleaning_done,
        "engineer": engineer_done,
        "select": select_done,
        "export": select_done or engineer_done or cleaning_done,
    }

    accessible = {
        "upload": True,
        "profile": has_dataset,
        "quality": has_dataset and profile_done,
        "cleaning": has_dataset and quality_done,
        "engineer": has_dataset and bool(pipeline.get("phase5_ready")),
        "select": has_dataset and bool(pipeline.get("phase6_ready")),
        "export": has_dataset
        and (
            bool(pipeline.get("cleaned_available"))
            or bool(pipeline.get("feature_engineered_available"))
            or bool(pipeline.get("feature_selected_available"))
            or cleaning_done
            or engineer_done
            or select_done
        ),
    }

    current_stage = active_nav if active_nav in STAGE_ORDER else "upload"
    if has_dataset and active_nav in STAGE_ORDER:
        for sid in STAGE_ORDER:
            if not completed.get(sid, False):
                current_stage = sid
                break
        else:
            current_stage = "export"

    filename = None
    if has_dataset:
        filename = (st.session_state["upload_payload"].get("dataset") or {}).get("filename")

    return PipelineContext(
        has_dataset=has_dataset,
        dataset_id=dataset_id,
        filename=filename,
        pipeline=pipeline,
        active_nav=active_nav,
        completed=completed,
        accessible=accessible,
        current_stage=current_stage,
        profile_done=profile_done,
        quality_done=quality_done,
        cleaning_done=cleaning_done,
        engineer_done=engineer_done,
        select_done=select_done,
    )


def navigate_to(stage: str) -> None:
    st.session_state["nav_page"] = stage
    st.rerun()


def reset_dataset(*, go_to_upload: bool = True) -> None:
    """Clear dataset and all derived pipeline session state."""
    for key in DATASET_SESSION_KEYS:
        st.session_state.pop(key, None)
    st.session_state["uploader_generation"] = (
        int(st.session_state.get("uploader_generation", 0)) + 1
    )
    if go_to_upload:
        st.session_state["nav_page"] = "upload"
    st.rerun()


def clamp_navigation(ctx: PipelineContext) -> str:
    """Redirect invalid nav targets to the nearest valid stage."""
    nav = ctx.active_nav
    if nav == "home":
        return nav
    if not ctx.has_dataset and nav != "upload":
        return "upload"
    if nav in STAGE_ORDER and not ctx.accessible.get(nav, False):
        # Walk backward to find last accessible workflow stage
        idx = STAGE_ORDER.index(nav)
        for i in range(idx - 1, -1, -1):
            sid = STAGE_ORDER[i]
            if ctx.accessible.get(sid):
                return sid
        return "upload"
    return nav


def next_stage(stage: str) -> str | None:
    if stage not in STAGE_ORDER:
        return None
    idx = STAGE_ORDER.index(stage)
    if idx + 1 < len(STAGE_ORDER):
        return STAGE_ORDER[idx + 1]
    return None
