"""Reusable shell components: header, sidebar, progress, page chrome."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.pipeline_state import (
    STAGE_LABELS,
    STAGE_ORDER,
    PipelineContext,
    build_pipeline_context,
    navigate_to,
    next_stage,
    reset_dataset,
)

WORKFLOW_STEPS = [
    {"id": "upload", "label": "Upload", "desc": "Add your raw CSV dataset."},
    {"id": "profile", "label": "Profile", "desc": "Understand columns, distributions and relationships."},
    {"id": "quality", "label": "Data Quality", "desc": "Detect missing values, duplicates, and outliers."},
    {"id": "cleaning", "label": "Cleaning", "desc": "Apply appropriate data-cleaning strategies."},
    {"id": "engineer", "label": "Feature Engineering", "desc": "Review and select useful feature transformations."},
    {"id": "select", "label": "Feature Selection", "desc": "Identify the most useful features for modeling."},
    {"id": "export", "label": "Export", "desc": "Download your final ML-ready dataset."},
]

HOME_HOW_IT_WORKS = [
    ("01", "Upload", "Add your raw CSV dataset."),
    ("02", "Profile", "Understand columns, distributions and relationships."),
    ("03", "Quality", "Detect missing values, duplicates, outliers and other issues."),
    ("04", "Clean", "Apply appropriate data-cleaning strategies."),
    ("05", "Engineer", "Review and select useful feature transformations."),
    ("06", "Select", "Identify the most useful features for modeling."),
    ("07", "Export", "Download your final ML-ready dataset."),
]


def render_logo_mark(size: str = "md") -> str:
    px = "34px" if size == "md" else "28px"
    fs = "0.85rem" if size == "md" else "0.75rem"
    return (
        f'<div class="pw-logo-mark" style="width:{px};height:{px};font-size:{fs};">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<ellipse cx="12" cy="5" rx="7" ry="3" stroke="white" stroke-width="1.8"/>'
        '<path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5" stroke="white" stroke-width="1.8"/>'
        '<path d="M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" stroke="white" stroke-width="1.8"/>'
        "</svg></div>"
    )


def render_brand_block(*, collapsed: bool = False, compact: bool = False) -> None:
    if collapsed:
        st.markdown(
            f'<div class="pw-brand-block"><div class="pw-logo-row">{render_logo_mark("sm")}</div></div>',
            unsafe_allow_html=True,
        )
        return
    tag = "" if compact else '<p class="pw-logo-tag pw-hide-collapsed">Intelligent ML Data Preparation</p>'
    st.markdown(
        f"""
<div class="pw-brand-block">
  <div class="pw-logo-row">
    {render_logo_mark()}
    <p class="pw-logo-title pw-hide-collapsed">PrepWise AI</p>
  </div>
  {tag}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_top_header(ctx: PipelineContext) -> None:
    """Header: brand + interactive dataset controls (no fake Deploy)."""
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown(
            f"""
<div class="pw-topbar-brand">
  {render_logo_mark("sm")}
  <div>
    <p class="pw-topbar-title">PrepWise AI</p>
    <p class="pw-topbar-sub">Intelligent ML Data Preparation</p>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        if ctx.has_dataset and ctx.filename:
            with st.popover(f"Dataset: {ctx.filename}", use_container_width=True):
                st.markdown(f"**{ctx.filename}**")
                if ctx.dataset_id is not None:
                    rows = (st.session_state.get("upload_payload", {}).get("dataset") or {}).get("rows")
                    cols = (st.session_state.get("upload_payload", {}).get("dataset") or {}).get("columns")
                    if rows is not None and cols is not None:
                        st.caption(f"{rows:,} rows · {cols:,} columns")
                if st.button("Replace Dataset", key="hdr_replace_dataset", use_container_width=True):
                    reset_dataset(go_to_upload=True)
                if st.button(
                    "Remove Dataset",
                    key="hdr_remove_dataset",
                    use_container_width=True,
                    type="secondary",
                ):
                    reset_dataset(go_to_upload=True)
        else:
            st.markdown(
                '<div class="pw-dataset-chip pw-dataset-empty">No dataset loaded</div>',
                unsafe_allow_html=True,
            )


def render_page_title(title: str, subtitle: str = "") -> None:
    st.markdown(f'<p class="pw-page-title">{title}</p>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="pw-page-sub">{subtitle}</p>', unsafe_allow_html=True)


def render_action_area(
    ctx: PipelineContext,
    *,
    page: str,
    primary_label: str | None = None,
    primary_disabled: bool = False,
    primary_key: str | None = None,
    show_continue: bool = True,
    completion_message: str | None = None,
    helper_text: str | None = None,
) -> bool:
    """
    Single contextual action area at page bottom.
    Returns True if primary in-page action button was clicked.
    """
    live = build_pipeline_context(active_nav=ctx.active_nav, pipeline=ctx.pipeline)
    clicked = False
    st.markdown('<div class="pw-action-area">', unsafe_allow_html=True)

    if completion_message:
        st.markdown(
            f'<p class="pw-status-ok">{completion_message}</p>',
            unsafe_allow_html=True,
        )
    if helper_text:
        st.caption(helper_text)

    cols = st.columns([2, 1])
    with cols[1]:
        if primary_label and primary_key:
            if st.button(
                primary_label,
                type="primary",
                key=primary_key,
                use_container_width=True,
                disabled=primary_disabled,
            ):
                clicked = True
        elif show_continue and page in STAGE_ORDER:
            nxt = next_stage(page)
            if nxt:
                label = f"Continue to {STAGE_LABELS.get(nxt, nxt)} →"
                can_advance = live.completed.get(page, False) or page == "upload"
                if can_advance and st.button(
                    label,
                    type="primary",
                    key=f"continue_{page}",
                    use_container_width=True,
                ):
                    navigate_to(nxt)

    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


def render_pipeline_progress(ctx: PipelineContext) -> None:
    cards: list[str] = []
    for idx, step in enumerate(WORKFLOW_STEPS, start=1):
        sid = step["id"]
        done = ctx.completed.get(sid, False)
        is_current = sid == ctx.current_stage or sid == ctx.active_nav
        if done and not is_current:
            css = "done"
        elif is_current:
            css = "current"
        else:
            css = ""
        cards.append(
            f'<div class="pw-wf-step {css}">'
            f'<div class="pw-wf-num">{idx}</div>'
            f'<p class="pw-wf-title">{step["label"]}</p>'
            f'<p class="pw-wf-desc">{step["desc"]}</p>'
            f"</div>"
        )
    st.markdown(
        f'<div class="pw-home-card pw-progress-strip">'
        f'<div class="pw-workflow-grid">{"".join(cards)}</div></div>',
        unsafe_allow_html=True,
    )


def render_home_hero(ctx: PipelineContext) -> None:
    st.markdown(
        """
<div class="pw-hero home-hero">
  <div class="pw-hero-inner">
    <h1 class="pw-hero-title">PrepWise AI</h1>
    <p class="pw-hero-sub">Intelligent ML Data Preparation</p>
    <p class="pw-hero-desc pw-hero-lead">
      Turn raw datasets into clean, reliable, machine-learning-ready data —
      without manually performing every preprocessing step.
    </p>
    <p class="pw-hero-support">
      Analyze, clean, engineer, and prepare your data for machine learning —
      all in one workflow.
    </p>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="pw-hero-cta-marker"></div>', unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.15, 1])
    with center:
        if ctx.has_dataset and ctx.filename:
            label = f"Continue with {ctx.filename} →"
            if st.button(
                label,
                type="primary",
                key="home_continue_dataset",
                use_container_width=True,
            ):
                target = ctx.current_stage if ctx.current_stage != "upload" else "profile"
                navigate_to(target)
        else:
            if st.button(
                "Upload Dataset →",
                type="primary",
                key="home_upload_primary",
                use_container_width=True,
            ):
                navigate_to("upload")


def render_home_how_it_works() -> None:
    items = []
    for num, title, desc in HOME_HOW_IT_WORKS:
        items.append(
            f'<div class="pw-how-item">'
            f'<span class="pw-how-num">{num}</span>'
            f'<div><p class="pw-how-title">{title}</p>'
            f'<p class="pw-how-desc">{desc}</p></div></div>'
        )
    st.markdown(
        f"""
<div class="pw-home-card">
  <p class="pw-home-card-title">How PrepWise Works</p>
  <div class="pw-how-grid">{"".join(items)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_stepper(ctx: PipelineContext, collapsed: bool) -> None:
    parts = ['<div class="pw-stepper">']
    for idx, step in enumerate(WORKFLOW_STEPS):
        sid = step["id"]
        done = ctx.completed.get(sid, False)
        is_current = sid == ctx.active_nav
        if done and not is_current:
            state = "done"
        elif is_current:
            state = "current"
        else:
            state = ""
        label_html = (
            ""
            if collapsed
            else f'<span class="pw-stepper-label pw-hide-collapsed">{step["label"]}</span>'
        )
        line = (
            '<div class="pw-stepper-line"></div>'
            if idx < len(WORKFLOW_STEPS) - 1
            else ""
        )
        parts.append(
            f'<div class="pw-stepper-item {state}">'
            f'<div class="pw-stepper-rail">'
            f'<div class="pw-stepper-dot {state}"></div>{line}'
            f"</div>{label_html}</div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_workflow_sidebar(ctx: PipelineContext) -> PipelineContext:
    """Render sidebar nav; may update active_nav from button clicks."""
    active_nav = ctx.active_nav
    collapsed = bool(st.session_state.get("sidebar_collapsed", False))

    with st.sidebar:
        render_brand_block(collapsed=collapsed)

        if not collapsed:
            st.markdown('<p class="pw-nav-label pw-hide-collapsed">Home</p>', unsafe_allow_html=True)
        home_label = "⌂" if collapsed else "⌂  Home"
        if st.button(
            home_label,
            key="nav_home",
            use_container_width=True,
            type="primary" if active_nav == "home" else "secondary",
        ):
            active_nav = "home"

        if not collapsed:
            st.markdown(
                '<p class="pw-nav-label pw-hide-collapsed">Workflow</p>',
                unsafe_allow_html=True,
            )
            _render_sidebar_stepper(ctx, collapsed=False)

        short = {
            "upload": "↑",
            "profile": "◎",
            "quality": "◇",
            "cleaning": "✦",
            "engineer": "⚙",
            "select": "☰",
            "export": "↓",
        }

        for step in WORKFLOW_STEPS:
            sid = step["id"]
            done = ctx.completed.get(sid, False)
            is_current = sid == ctx.active_nav
            if done and sid != active_nav:
                mark = "✓"
            elif is_current:
                mark = "●"
            else:
                mark = "○"
            if collapsed:
                label = short[sid]
            else:
                label = f"{mark}  {step['label']}"
            disabled = not ctx.accessible.get(sid, False) and sid != "upload"
            if not ctx.has_dataset and sid != "upload":
                disabled = True
            btn_type = "primary" if active_nav == sid else "secondary"
            if st.button(
                label,
                key=f"nav_{sid}",
                use_container_width=True,
                type=btn_type,
                disabled=disabled,
            ):
                active_nav = sid

        if not collapsed:
            st.markdown(
                """
<div class="pw-sidebar-help pw-hide-collapsed">
  <strong>Need Help?</strong>
  Use the workflow steps in order. Replace the dataset from the header menu anytime.
</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        toggle_label = "»" if collapsed else "«  Collapse"
        if st.button(toggle_label, key="sidebar_toggle", use_container_width=True):
            st.session_state["sidebar_collapsed"] = not collapsed
            st.rerun()

    return PipelineContext(
        has_dataset=ctx.has_dataset,
        dataset_id=ctx.dataset_id,
        filename=ctx.filename,
        pipeline=ctx.pipeline,
        active_nav=active_nav,
        completed=ctx.completed,
        accessible=ctx.accessible,
        current_stage=ctx.current_stage,
        profile_done=ctx.profile_done,
        quality_done=ctx.quality_done,
        cleaning_done=ctx.cleaning_done,
        engineer_done=ctx.engineer_done,
        select_done=ctx.select_done,
    )
