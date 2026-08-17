"""PrepWise AI — Streamlit frontend (navy/white redesign).

UI presentation only. Backend APIs and processing logic are unchanged.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd
import requests
import streamlit as st

from ui.cleaning_dashboard import (
    build_cleaning_config_from_ui,
    render_before_after,
    render_cleaned_preview,
    render_cleaning_log,
    render_cleaning_summary,
    render_download_button,
)
from ui.feature_engineering_dashboard import (
    render_feature_candidate_selector,
    render_featured_download_button,
    render_phase5_feature_engineering_result,
)
from ui.feature_selection_dashboard import (
    render_feature_selection_dashboard,
    render_selected_download_buttons,
)
from ui.profile_dashboard import render_profile_dashboard
from ui.quality_dashboard import render_quality_dashboard
from ui.pipeline_state import (
    DATASET_SESSION_KEYS,
    PipelineContext,
    build_pipeline_context,
    clamp_navigation,
    navigate_to,
    reset_dataset,
)
from ui.shell import (
    render_action_area,
    render_home_hero,
    render_page_title,
    render_pipeline_progress,
    render_top_header,
    render_workflow_sidebar,
)
from ui.theme import apply_global_styles

BACKEND_URL = os.getenv("AUTOPREP_BACKEND_URL", "http://127.0.0.1:8000")
UPLOAD_ENDPOINT = f"{BACKEND_URL}/api/v1/datasets/upload"

SESSION_DERIVED_KEYS: tuple[str, ...] = (
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


def _clear_derived_pipeline_state() -> None:
    for key in SESSION_DERIVED_KEYS:
        st.session_state.pop(key, None)


def _profile_endpoint(dataset_id: int) -> str:
    return f"{BACKEND_URL}/api/v1/datasets/{dataset_id}/profile"


def _quality_endpoint(dataset_id: int) -> str:
    return f"{BACKEND_URL}/api/v1/datasets/{dataset_id}/quality"


def _clean_preview_endpoint(dataset_id: int) -> str:
    return f"{BACKEND_URL}/api/v1/datasets/{dataset_id}/clean/preview"


def _clean_apply_endpoint(dataset_id: int) -> str:
    return f"{BACKEND_URL}/api/v1/datasets/{dataset_id}/clean"


def _feature_candidates_endpoint(dataset_id: int) -> str:
    return f"{BACKEND_URL}/api/v1/datasets/{dataset_id}/feature-engineering/candidates"


def _phase5_fe_endpoint(dataset_id: int) -> str:
    return f"{BACKEND_URL}/api/v1/datasets/{dataset_id}/feature-engineering"


def _pipeline_status_endpoint(dataset_id: int) -> str:
    return f"{BACKEND_URL}/api/v1/datasets/{dataset_id}/pipeline-status"


def _feature_selection_analyze_endpoint(dataset_id: int) -> str:
    return f"{BACKEND_URL}/api/v1/datasets/{dataset_id}/feature-selection/analyze"


def _feature_selection_apply_endpoint(dataset_id: int) -> str:
    return f"{BACKEND_URL}/api/v1/datasets/{dataset_id}/feature-selection/apply"


def _fetch_pipeline_status(dataset_id: int) -> dict | None:
    try:
        response = requests.get(_pipeline_status_endpoint(dataset_id), timeout=30)
    except requests.exceptions.RequestException:
        return None
    if response.status_code != 200:
        return None
    return response.json()


ENCODING_CANDIDATES: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "latin-1",
    "cp1252",
)

_TARGET_NAME_HINT = re.compile(
    r"(target|label|class|outcome|churn|purchased|purchase|fraud|survived|"
    r"default|approved|success|failure|response|y_true|y_label)",
    re.IGNORECASE,
)


def _format_upload_time(iso_value: str) -> str:
    try:
        normalized = iso_value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return dt.strftime("%b %d, %Y · %I:%M %p")
    except (TypeError, ValueError):
        return str(iso_value)


def _detect_encoding(raw: bytes) -> str:
    for encoding in ENCODING_CANDIDATES:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "Unknown"


def _load_dataframe(raw: bytes, encoding: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(raw), encoding=encoding)


def _upload_cache_key(uploaded_file: Any) -> str:
    return f"{uploaded_file.name}:{uploaded_file.size}"



def _overview_column_counts(df: pd.DataFrame) -> dict[str, int]:
    numerical = categorical = date_cols = text = 0
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            categorical += 1
            continue
        if pd.api.types.is_numeric_dtype(series):
            numerical += 1
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols += 1
            continue
        non_null = series.dropna()
        if non_null.empty:
            categorical += 1
            continue
        sample = non_null.head(min(50, len(non_null)))
        parsed = pd.to_datetime(sample, errors="coerce")
        if float(parsed.notna().mean()) >= 0.8:
            date_cols += 1
            continue
        as_str = sample.astype(str)
        mean_len = float(as_str.str.len().mean()) if len(as_str) else 0.0
        nunique = int(non_null.nunique())
        if mean_len >= 40 or (len(df) and nunique / max(len(df), 1) > 0.5 and mean_len >= 20):
            text += 1
        else:
            categorical += 1
    return {
        "numerical": numerical,
        "categorical": categorical,
        "date": date_cols,
        "text": text,
    }


def _overview_target_hint(df: pd.DataFrame) -> tuple[str | None, str | None]:
    candidates = [str(c) for c in df.columns if _TARGET_NAME_HINT.search(str(c))]
    if not candidates:
        return None, None
    priority = ("churn", "target", "label", "class", "outcome")
    target = candidates[0]
    for key in priority:
        for c in candidates:
            if key in c.lower().replace(" ", "_"):
                target = c
                break
        else:
            continue
        break
    series = df[target]
    non_null = series.dropna()
    nunique = int(non_null.nunique()) if len(non_null) else 0
    if nunique == 2:
        problem = "Binary Classification"
    elif nunique <= 20:
        problem = "Classification"
    elif pd.api.types.is_numeric_dtype(non_null) and nunique > 20:
        problem = "Regression"
    else:
        problem = "Classification"
    return target, problem


def render_dataset_overview(dataset: dict, df: pd.DataFrame) -> None:
    counts = _overview_column_counts(df)
    target, problem = _overview_target_hint(df)
    with st.container(border=True):
        st.markdown("##### Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{dataset['rows']:,}")
        c2.metric("Columns", f"{dataset['columns']:,}")
        c3.metric("Numerical", f"{counts['numerical']:,}")
        c4.metric("Categorical", f"{counts['categorical']:,}")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Date", f"{counts['date']:,}")
        d2.metric("Text", f"{counts['text']:,}")
        d3.metric("Target", target or "—")
        d4.metric("Problem", problem or "—")
        if target:
            st.caption(
                f"Possible target: `{target}` · {problem}. "
                "Confirm during profiling — name-based hint only."
            )


def _fetch_upload(uploaded_file: Any) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            UPLOAD_ENDPOINT,
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv",
                )
            },
            timeout=60,
        )
    except requests.exceptions.RequestException:
        return None, (
            f"Could not reach the backend. Make sure the FastAPI server is "
            f"running at {BACKEND_URL}."
        )
    if response.status_code == 201:
        return response.json(), None
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return None, f"Upload failed: {detail}"


def _fetch_profile(dataset_id: int) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(_profile_endpoint(dataset_id), timeout=120)
    except requests.exceptions.RequestException:
        return None, f"Could not reach the backend at {BACKEND_URL}."
    if response.status_code == 200:
        return response.json(), None
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return None, f"Profiling failed: {detail}"


def _fetch_quality(dataset_id: int) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(_quality_endpoint(dataset_id), timeout=120)
    except requests.exceptions.RequestException:
        return None, f"Could not reach the backend at {BACKEND_URL}."
    if response.status_code == 200:
        return response.json(), None
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return None, f"Quality assessment failed: {detail}"


def _post_json(url: str, payload: dict[str, Any]) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(url, json=payload, timeout=180)
    except requests.exceptions.RequestException:
        return None, f"Could not reach the backend at {BACKEND_URL}."
    if response.status_code == 200:
        return response.json(), None
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return None, str(detail)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _quality_has_issues(report: dict[str, Any]) -> bool:
    missing = report.get("missing_values") or []
    outliers = report.get("outliers") or []
    dup_rows = report.get("duplicate_rows") or {}
    constants = report.get("constant_columns") or []
    dup_cols = report.get("duplicate_columns") or []
    return bool(
        missing
        or outliers
        or constants
        or dup_cols
        or int(dup_rows.get("count") or 0) > 0
    )


def page_home(ctx: PipelineContext) -> None:
    render_home_hero(ctx)
    if ctx.has_dataset:
        render_pipeline_progress(ctx)


def page_upload(ctx: PipelineContext) -> None:
    render_page_title(
        "Upload Dataset",
        "Start by uploading the dataset you want to prepare.",
    )

    gen = int(st.session_state.get("uploader_generation", 0))
    uploaded_file = st.file_uploader(
        "Drag and drop a CSV file, or browse",
        type=["csv"],
        help="Supported format: CSV",
        key=f"main_uploader_{gen}",
    )

    if uploaded_file is None and ctx.has_dataset:
        st.markdown(
            f'<p class="pw-status-ok">✓ {ctx.filename} is loaded.</p>',
            unsafe_allow_html=True,
        )
        st.caption("Use the header menu to replace this dataset, or upload a new CSV below.")
        render_action_area(
            ctx,
            page="upload",
            show_continue=True,
            helper_text="Continue profiling or pick up where you left off.",
        )
        return

    if uploaded_file is None:
        st.info("Upload a CSV dataset to begin.")
        return

    cache_key = _upload_cache_key(uploaded_file)
    if st.session_state.get("upload_cache_key") != cache_key:
        with st.spinner("Uploading your dataset..."):
            payload, error = _fetch_upload(uploaded_file)
        if error:
            st.error(
                f"Unable to process this file.\n\n{error}\n\n"
                "Choose another CSV file and try again."
            )
            return
        st.session_state["upload_cache_key"] = cache_key
        st.session_state["upload_payload"] = payload
        st.session_state["uploaded_file_name"] = uploaded_file.name
        st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()
        _clear_derived_pipeline_state()
        st.session_state["nav_page"] = "profile"
        st.session_state["auto_run_profile"] = True
        st.success("Dataset uploaded successfully. Opening Profile...")
        st.rerun()

    st.session_state["nav_page"] = "profile"
    st.session_state["auto_run_profile"] = True
    st.rerun()


def _ensure_profile(dataset_id: int) -> bool:
    """Run profiling if needed; return True when profile is ready."""
    if (
        st.session_state.get("profile_dataset_id") == dataset_id
        and st.session_state.get("profile_payload")
    ):
        return True
    with st.spinner("Analyzing dataset...\nProfiling columns...\nChecking distributions..."):
        profile, error = _fetch_profile(dataset_id)
    if error:
        st.error(error)
        return False
    st.session_state["profile_payload"] = profile
    st.session_state["profile_dataset_id"] = dataset_id
    return True


def page_profile(
    ctx: PipelineContext,
    dataset: dict,
    df: pd.DataFrame,
    preview: list,
) -> None:
    render_page_title(
        "Profile",
        "Understand your dataset before making changes.",
    )
    render_pipeline_progress(ctx)

    dataset_id = dataset["id"]
    if st.session_state.pop("auto_run_profile", False) or not ctx.profile_done:
        _ensure_profile(dataset_id)

    profile_ready = (
        st.session_state.get("profile_dataset_id") == dataset_id
        and "profile_payload" in st.session_state
    )
    if profile_ready:
        render_profile_dashboard(
            st.session_state["profile_payload"],
            preview=preview,
            df=df,
        )
        render_action_area(
            ctx,
            page="profile",
            completion_message="Profile analysis complete ✓",
            helper_text="You can now review detected data-quality issues.",
            show_continue=True,
        )
    else:
        st.info("Profiling will start automatically when you open this page.")


def _ensure_quality(dataset_id: int) -> bool:
    if (
        st.session_state.get("quality_dataset_id") == dataset_id
        and st.session_state.get("quality_payload")
    ):
        return True
    with st.spinner("Detecting quality issues...\nChecking missing values...\nScanning duplicates..."):
        report, error = _fetch_quality(dataset_id)
    if error:
        st.error(error)
        return False
    st.session_state["quality_payload"] = report
    st.session_state["quality_dataset_id"] = dataset_id
    return True


def page_quality(ctx: PipelineContext, dataset: dict) -> None:
    render_page_title(
        "Data Quality",
        "Identify issues that may affect your machine-learning workflow.",
    )
    render_pipeline_progress(ctx)

    dataset_id = dataset["id"]
    if not ctx.profile_done:
        st.warning("Complete Profile before assessing data quality.")
        return

    if st.session_state.pop("auto_run_quality", False) or not ctx.quality_done:
        _ensure_quality(dataset_id)

    if (
        st.session_state.get("quality_dataset_id") == dataset_id
        and "quality_payload" in st.session_state
    ):
        report = st.session_state["quality_payload"]
        render_quality_dashboard(report)
        has_issues = _quality_has_issues(report)
        nxt_label = (
            "Review Cleaning Plan →" if has_issues else "Continue to Cleaning →"
        )
        if render_action_area(
            ctx,
            page="quality",
            completion_message="Quality assessment complete ✓",
            primary_label=nxt_label,
            primary_key="quality_forward",
            show_continue=False,
        ):
            navigate_to("cleaning")
    else:
        if st.button("Assess Data Quality", type="primary", key="assess_quality_manual"):
            if _ensure_quality(dataset_id):
                st.session_state["auto_run_quality"] = False
                st.rerun()



def page_cleaning(ctx: PipelineContext, dataset: dict, pipeline: dict | None) -> None:
    render_page_title(
        "Cleaning",
        "Fix detected problems using the recommended cleaning plan.",
    )
    render_pipeline_progress(ctx)
    dataset_id = dataset["id"]
    config = build_cleaning_config_from_ui()

    session_applied = (
        st.session_state.get("clean_result_dataset_id") == dataset_id
        and "clean_result" in st.session_state
    )

    # --- Preview (optional secondary) ---
    preview_clicked = st.button(
        "Preview Cleaning Plan",
        key="preview_clean_btn",
        type="secondary",
    )
    if preview_clicked:
        with st.spinner("Running cleaning dry-run..."):
            preview, error = _post_json(_clean_preview_endpoint(dataset_id), config)
        if error:
            st.error(f"Cleaning preview failed: {error}")
        else:
            st.session_state["clean_preview"] = preview
            st.session_state["clean_preview_dataset_id"] = dataset_id

    if (
        st.session_state.get("clean_preview_dataset_id") == dataset_id
        and "clean_preview" in st.session_state
        and not session_applied
    ):
        preview = st.session_state["clean_preview"]
        st.info("Dry-run complete — no files were written.")
        render_cleaning_summary(preview["summary"])
        render_cleaning_log(preview.get("planned_log", []))

    # --- Apply Cleaning (always available until applied this session) ---
    if not session_applied:
        st.markdown("")
        apply_clicked = st.button(
            "Apply Cleaning",
            type="primary",
            use_container_width=True,
            key="apply_clean_btn",
        )
        if apply_clicked:
            with st.spinner("Applying cleaning and saving cleaned copy..."):
                result, error = _post_json(_clean_apply_endpoint(dataset_id), config)
            if error:
                st.error(f"Cleaning failed: {error}")
            else:
                st.session_state["clean_result"] = result
                st.session_state["clean_result_dataset_id"] = dataset_id
                st.session_state.pop("clean_preview", None)
                st.session_state.pop("clean_preview_dataset_id", None)
                st.success("Cleaning completed successfully.")
                st.rerun()
        render_action_area(
            ctx,
            page="cleaning",
            helper_text="Configure options above, then apply cleaning to continue.",
            show_continue=False,
        )
        return

    # --- Results after apply ---
    result = st.session_state["clean_result"]
    render_cleaning_summary(result["summary"])
    render_before_after(result["before"], result["after"])
    render_cleaned_preview(result.get("preview", []))
    render_cleaning_log(result.get("cleaning_log", []))
    if render_action_area(
        ctx,
        page="cleaning",
        completion_message="Cleaning complete ✓",
        primary_label="Continue to Feature Engineering →",
        primary_key="cleaning_forward",
        show_continue=False,
    ):
        navigate_to("engineer")


def page_engineer(ctx: PipelineContext, dataset: dict, pipeline: dict | None) -> None:
    render_page_title(
        "Feature Engineering",
        "Review recommendations, choose what to create, then generate only those features.",
    )
    render_pipeline_progress(ctx)
    dataset_id = dataset["id"]
    phase5_ready = bool(pipeline and pipeline.get("phase5_ready"))
    if not phase5_ready:
        st.warning("Complete Cleaning before Feature Engineering.")
        return

    has_candidates = (
        st.session_state.get("fe_candidates_dataset_id") == dataset_id
        and "fe_candidates_payload" in st.session_state
    )
    fe_done = (
        st.session_state.get("phase5_fe_dataset_id") == dataset_id
        and "phase5_fe_payload" in st.session_state
    )

    if not has_candidates and not fe_done:
        if st.button(
            "Analyze & Recommend Features",
            use_container_width=True,
            key="feature_candidates_btn",
            type="primary",
        ):
            with st.spinner("Analyzing feature opportunities..."):
                try:
                    response = requests.post(
                        _feature_candidates_endpoint(dataset_id),
                        timeout=300,
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach backend: {exc}")
                    return
                if response.status_code != 200:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.error(f"Feature recommendation failed: {detail}")
                    return
                st.session_state["fe_candidates_payload"] = response.json()
                st.session_state["fe_candidates_dataset_id"] = dataset_id
                st.session_state.pop("phase5_fe_payload", None)
                st.session_state.pop("phase5_fe_dataset_id", None)
                st.rerun()

    selected_ids: list[str] = []
    if has_candidates and not fe_done:
        payload = st.session_state["fe_candidates_payload"]
        recs = payload.get("recommendations") or payload.get("candidates") or []
        if not recs:
            st.info(
                "Your current features appear sufficient. "
                "No additional feature engineering is required."
            )
        else:
            selected_ids = render_feature_candidate_selector(payload)
            st.caption("Leave selection empty to pass cleaned columns through unchanged.")
        if st.button(
            "Generate Selected Features",
            use_container_width=True,
            key="phase5_fe_btn",
            type="primary",
        ):
            if not selected_ids and recs:
                selected_ids = list(
                    st.session_state.get("fe_candidate_multiselect", []) or []
                )
            with st.spinner("Generating selected features..."):
                try:
                    response = requests.post(
                        _phase5_fe_endpoint(dataset_id),
                        json={"selected_feature_ids": selected_ids},
                        timeout=300,
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach backend: {exc}")
                    return
                if response.status_code != 200:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.error(f"Feature engineering failed: {detail}")
                    return
                st.session_state["phase5_fe_payload"] = response.json()
                st.session_state["phase5_fe_dataset_id"] = dataset_id
                st.rerun()

    if fe_done:
        result = st.session_state["phase5_fe_payload"]
        render_phase5_feature_engineering_result(result)
        if render_action_area(
            ctx,
            page="engineer",
            completion_message="Feature engineering complete ✓",
            primary_label="Continue to Feature Selection →",
            primary_key="engineer_forward",
            show_continue=False,
        ):
            navigate_to("select")


def page_select(ctx: PipelineContext, dataset: dict, pipeline: dict | None) -> None:
    render_page_title(
        "Feature Selection",
        "Select the final features that should remain in your ML-ready dataset.",
    )
    render_pipeline_progress(ctx)
    dataset_id = dataset["id"]
    phase6_ready = bool(pipeline and pipeline.get("phase6_ready"))
    if not phase6_ready:
        st.warning("Complete Feature Engineering before Feature Selection.")
        return

    target_override = st.text_input(
        "Optional target column (leave blank for auto-detect)",
        key=f"fs_target_{dataset_id}",
    )
    body: dict[str, Any] = {}
    if target_override and target_override.strip():
        body["target_column"] = target_override.strip()

    has_analysis = (
        st.session_state.get("fs_analyze_dataset_id") == dataset_id
        and "fs_analyze_payload" in st.session_state
    )
    fs_done = (
        st.session_state.get("fs_apply_dataset_id") == dataset_id
        and "fs_apply_payload" in st.session_state
    )

    if not has_analysis and not fs_done:
        if st.button(
            "Analyze Features",
            use_container_width=True,
            key="feature_sel_analyze_btn",
            type="primary",
        ):
            with st.spinner("Analyzing features for KEEP / REVIEW / REMOVE..."):
                try:
                    response = requests.post(
                        _feature_selection_analyze_endpoint(dataset_id),
                        json=body,
                        timeout=300,
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach backend: {exc}")
                    return
                if response.status_code != 200:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.error(f"Feature selection analysis failed: {detail}")
                    return
                st.session_state["fs_analyze_payload"] = response.json()
                st.session_state["fs_analyze_dataset_id"] = dataset_id
                st.session_state.pop("fs_apply_payload", None)
                st.session_state.pop("fs_apply_dataset_id", None)
                st.rerun()

    also_remove: list[str] = []
    if has_analysis and not fs_done:
        analyze_payload = st.session_state["fs_analyze_payload"]
        render_feature_selection_dashboard(analyze_payload)
        review_features = [
            d.get("feature")
            for d in (analyze_payload.get("decisions") or [])
            if d.get("decision") == "REVIEW" and d.get("feature")
        ]
        if review_features:
            also_remove = st.multiselect(
                "Also remove REVIEW features (optional — REVIEW kept by default)",
                options=review_features,
                key=f"fs_also_remove_{dataset_id}",
            )
        if st.button(
            "Apply Feature Selection",
            type="primary",
            use_container_width=True,
            key="feature_sel_apply_btn",
        ):
            apply_body = {
                **body,
                "apply_recommended": True,
                "also_remove": also_remove
                if also_remove
                else list(st.session_state.get(f"fs_also_remove_{dataset_id}", []) or []),
            }
            with st.spinner("Applying feature selection..."):
                try:
                    response = requests.post(
                        _feature_selection_apply_endpoint(dataset_id),
                        json=apply_body,
                        timeout=300,
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach backend: {exc}")
                    return
                if response.status_code != 200:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    st.error(f"Feature selection apply failed: {detail}")
                    return
                st.session_state["fs_apply_payload"] = response.json()
                st.session_state["fs_apply_dataset_id"] = dataset_id
                st.rerun()

    if fs_done:
        result = st.session_state["fs_apply_payload"]
        render_feature_selection_dashboard(result)
        render_selected_download_buttons(
            BACKEND_URL,
            dataset_id,
            result.get("selected_filename"),
        )
        if render_action_area(
            ctx,
            page="select",
            completion_message="Feature selection complete ✓",
            primary_label="Continue to Export →",
            primary_key="select_forward",
            show_continue=False,
        ):
            navigate_to("export")


def page_export(ctx: PipelineContext, dataset: dict, df: pd.DataFrame, pipeline: dict | None) -> None:
    render_page_title(
        "Your Dataset Is Ready",
        "Your prepared dataset is ready for machine learning.",
    )
    render_pipeline_progress(ctx)

    pipeline = pipeline or {}
    exports = pipeline.get("exports") or {}
    target, problem = _overview_target_hint(df)

    before_cols = dataset.get("columns", len(df.columns))
    clean = st.session_state.get("clean_result") or {}
    fe = st.session_state.get("phase5_fe_payload") or {}
    fs = st.session_state.get("fs_apply_payload") or {}

    final_rows = (
        (fs.get("after") or {}).get("rows")
        or (fe.get("after") or {}).get("rows")
        or (clean.get("after") or {}).get("rows")
        or dataset.get("rows")
    )
    final_cols = (
        (fs.get("after") or {}).get("columns")
        or (fe.get("after") or {}).get("columns")
        or (clean.get("after") or {}).get("columns")
        or before_cols
    )

    with st.container(border=True):
        st.markdown("##### Final Dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{final_rows:,}" if final_rows is not None else "—")
        c2.metric("Features", f"{final_cols:,}" if final_cols is not None else "—")
        c3.metric("Target", target or "—")
        c4.metric("Problem", problem or "—")

    steps = [
        ("Uploaded", ctx.completed.get("upload", False)),
        ("Profiled", ctx.profile_done),
        ("Quality checked", ctx.quality_done),
        ("Cleaned", ctx.cleaning_done),
        ("Features engineered", ctx.engineer_done),
        ("Features selected", ctx.select_done),
        ("Ready to export", ctx.select_done or ctx.engineer_done or ctx.cleaning_done),
    ]
    summary_html = " · ".join(
        f"{'✓' if done else '○'} {label}" for label, done in steps
    )
    st.markdown(f'<p class="pw-status-ok">{summary_html}</p>', unsafe_allow_html=True)

    if exports.get("feature_selected") or exports.get("feature_engineered") or exports.get("cleaned"):
        st.markdown(
            '<p class="pw-status-ok">✓ ML-ready dataset generated.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Complete earlier pipeline stages to unlock exports.")

    dataset_id = dataset["id"]
    st.markdown("##### Download")
    if fs.get("selected_filename"):
        render_selected_download_buttons(
            BACKEND_URL,
            dataset_id,
            fs.get("selected_filename"),
        )
    elif fe.get("featured_filename"):
        render_featured_download_button(
            BACKEND_URL,
            dataset_id,
            fe["featured_filename"],
            full_phase5=True,
        )
    elif clean.get("cleaned_filename"):
        render_download_button(
            BACKEND_URL,
            dataset_id,
            clean.get("cleaned_filename", "cleaned_dataset.csv"),
        )

    if st.button("Replace Dataset", key="export_replace_dataset", type="secondary"):
        reset_dataset(go_to_upload=True)




def main() -> None:
    st.set_page_config(
        page_title="PrepWise AI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if "sidebar_collapsed" not in st.session_state:
        st.session_state["sidebar_collapsed"] = False
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "home"
    if "uploader_generation" not in st.session_state:
        st.session_state["uploader_generation"] = 0

    apply_global_styles(sidebar_collapsed=bool(st.session_state["sidebar_collapsed"]))

    has_upload = "upload_payload" in st.session_state
    dataset = None
    df = None
    preview: list = []
    pipeline = None

    if has_upload:
        data = st.session_state["upload_payload"]
        dataset = data["dataset"]
        preview = data.get("preview") or []
        raw = st.session_state.get("uploaded_file_bytes")
        if raw:
            encoding = _detect_encoding(raw)
            try:
                df = _load_dataframe(
                    raw, encoding if encoding != "Unknown" else "utf-8"
                )
            except Exception:
                try:
                    df = _load_dataframe(raw, "latin-1")
                except Exception:
                    df = pd.DataFrame(preview)
        else:
            df = pd.DataFrame(preview)
        pipeline = _fetch_pipeline_status(dataset["id"])
        st.session_state["pipeline_status"] = pipeline

    active = st.session_state.get("nav_page", "home")
    ctx = build_pipeline_context(active_nav=active, pipeline=pipeline)
    active = clamp_navigation(ctx)
    st.session_state["nav_page"] = active
    ctx = build_pipeline_context(active_nav=active, pipeline=pipeline)

    ctx = render_workflow_sidebar(ctx)
    st.session_state["nav_page"] = ctx.active_nav
    clamped = clamp_navigation(ctx)
    if clamped != ctx.active_nav:
        st.session_state["nav_page"] = clamped
        st.rerun()
    ctx = build_pipeline_context(active_nav=clamped, pipeline=pipeline)

    render_top_header(ctx)

    if ctx.active_nav == "home":
        page_home(ctx)
    elif ctx.active_nav == "upload":
        page_upload(ctx)
    elif not has_upload or dataset is None or df is None:
        st.warning("No dataset loaded.")
        st.info("Upload a CSV dataset to begin profiling.")
        if st.button("Upload Dataset →", type="primary", key="empty_upload_cta"):
            navigate_to("upload")
    elif ctx.active_nav == "profile":
        page_profile(ctx, dataset, df, preview)
    elif ctx.active_nav == "quality":
        page_quality(ctx, dataset)
    elif ctx.active_nav == "cleaning":
        page_cleaning(ctx, dataset, pipeline)
    elif ctx.active_nav == "engineer":
        page_engineer(ctx, dataset, pipeline)
    elif ctx.active_nav == "select":
        page_select(ctx, dataset, pipeline)
    elif ctx.active_nav == "export":
        page_export(ctx, dataset, df, pipeline)
    else:
        page_home(ctx)

if __name__ == "__main__":
    main()
