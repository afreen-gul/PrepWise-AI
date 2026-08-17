"""Streamlit UI for Phase 6 feature selection."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st


def render_feature_selection_summary(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    with st.container(border=True):
        st.markdown("##### Feature Selection")
        st.caption(report.get("message") or "")
        st.caption(
            f"Dataset source: **{report.get('source', 'feature_engineered')}**"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Features", f"{summary.get('total_features', 0):,}")
        c2.metric("KEEP", f"{summary.get('keep', 0):,}")
        c3.metric("REVIEW", f"{summary.get('review', 0):,}")
        c4.metric("REMOVE", f"{summary.get('remove', 0):,}")

        target = summary.get("target_column")
        if target:
            st.info(
                f"Target detected: `{target}` "
                f"({summary.get('target_task') or 'task unknown'}). "
                "Target-aware scoring "
                + (
                    "was applied."
                    if summary.get("target_aware_applied")
                    else "was limited or skipped."
                )
            )
        else:
            st.warning(
                report.get("target_message")
                or "No target detected. Target-aware feature selection was skipped."
            )


def render_quality_table(report: dict[str, Any]) -> None:
    rows = report.get("quality_rows") or []
    with st.container(border=True):
        st.markdown("#### QUALITY ANALYSIS")
        if not rows:
            st.caption("No quality rows.")
            return
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": r.get("feature"),
                        "Type": r.get("semantic_type"),
                        "Missing %": r.get("missing_pct"),
                        "Unique %": r.get("unique_pct"),
                        "Constant": r.get("is_constant"),
                        "Near-constant": r.get("is_near_constant"),
                        "Identifier": r.get("is_identifier"),
                        "Duplicate": r.get("is_exact_duplicate"),
                        "Generated": r.get("is_generated"),
                        "Flags": ", ".join(r.get("quality_flags") or []),
                    }
                    for r in rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_correlation_table(report: dict[str, Any]) -> None:
    pairs = report.get("correlation_pairs") or []
    with st.container(border=True):
        st.markdown("#### CORRELATION ANALYSIS")
        st.caption("High-correlation pairs only (|r| ≥ threshold).")
        if not pairs:
            st.caption("No highly correlated numerical pairs detected.")
            return
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature A": p.get("feature_a"),
                        "Feature B": p.get("feature_b"),
                        "Correlation": p.get("correlation"),
                        "Recommendation": p.get("recommendation"),
                        "Reason": p.get("reason"),
                    }
                    for p in pairs
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_vif_table(report: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("#### MULTICOLLINEARITY (VIF)")
        if not report.get("vif_available", True):
            st.info(report.get("vif_message") or "VIF unavailable for this dataset.")
            return
        rows = report.get("vif_rows") or []
        if not rows:
            st.caption("No VIF rows.")
            return
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": r.get("feature"),
                        "VIF": r.get("vif") if r.get("vif") is not None else "N/A",
                        "Status": r.get("status"),
                        "Related Features": ", ".join(r.get("related_features") or []),
                        "Recommendation": r.get("recommendation"),
                    }
                    for r in rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_target_table(report: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("#### TARGET-AWARE ANALYSIS")
        scores = report.get("target_scores") or []
        if not scores:
            st.info(
                report.get("target_message")
                or "No target detected. Target-aware feature selection was skipped."
            )
            return
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": r.get("feature"),
                        "Target Type": r.get("target_type"),
                        "MI Score": r.get("mi_score"),
                        "Rank": r.get("rank"),
                        "Recommendation": r.get("recommendation"),
                        "Interpretation": r.get("interpretation"),
                    }
                    for r in scores
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_decision_table(report: dict[str, Any]) -> None:
    decisions = report.get("decisions") or []
    with st.container(border=True):
        st.markdown("#### FINAL FEATURE DECISIONS")
        if not decisions:
            st.caption("No decisions available.")
            return
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": d.get("feature"),
                        "Type": d.get("feature_type"),
                        "Status": d.get("status"),
                        "Missing %": d.get("missing_pct"),
                        "Unique %": d.get("unique_pct"),
                        "Correlation": d.get("correlation") or "N/A",
                        "VIF": d.get("vif") or "N/A",
                        "Target Score": d.get("target_score") or "N/A",
                        "Decision": d.get("decision"),
                        "Reason": d.get("reason"),
                    }
                    for d in decisions
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        explanations = report.get("explanations") or []
        if explanations:
            with st.expander("Beginner-friendly explanations", expanded=False):
                for exp in explanations:
                    st.markdown(f"- {exp}")

        with st.expander("Decision evidence details", expanded=False):
            for d in decisions:
                st.markdown(
                    f"**`{d.get('feature')}` → {d.get('decision')}** — {d.get('reason')}"
                )
                evidence = d.get("evidence") or []
                methods = d.get("methods") or []
                if evidence:
                    st.caption("Evidence: " + "; ".join(str(e) for e in evidence))
                if methods:
                    st.caption("Methods: " + ", ".join(methods))
                if d.get("is_generated"):
                    st.caption(
                        f"Generated from `{d.get('source_feature')}` via "
                        f"{d.get('transformation')}"
                    )
                st.divider()


def render_feature_selection_dashboard(report: dict[str, Any]) -> None:
    render_feature_selection_summary(report)
    render_quality_table(report)
    render_correlation_table(report)
    render_vif_table(report)
    render_target_table(report)
    render_decision_table(report)

    if report.get("applied"):
        before = report.get("before") or {}
        after = report.get("after") or {}
        st.success(
            "Feature-selected dataset created. "
            f"Rows {before.get('rows', 0):,} → {after.get('rows', 0):,} · "
            f"Columns {before.get('columns', 0):,} → {after.get('columns', 0):,}."
        )


def render_selected_download_buttons(
    backend_url: str,
    dataset_id: int,
    selected_filename: str | None = None,
) -> None:
    csv_name = selected_filename or "selected_dataset.csv"
    try:
        csv_resp = requests.get(
            f"{backend_url}/api/v1/datasets/{dataset_id}/feature-selection/download",
            timeout=60,
        )
        report_resp = requests.get(
            f"{backend_url}/api/v1/datasets/{dataset_id}/feature-selection/report/download",
            timeout=60,
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not download exports: {exc}")
        return

    c1, c2 = st.columns(2)
    with c1:
        if csv_resp.status_code == 200:
            st.download_button(
                "⬇ Download Feature-Selected Dataset",
                data=csv_resp.content,
                file_name=csv_name,
                mime="text/csv",
                use_container_width=True,
                key=f"dl_selected_{dataset_id}",
            )
        else:
            st.caption("Selected dataset not available for download yet.")
    with c2:
        if report_resp.status_code == 200:
            st.download_button(
                "⬇ Download Feature Selection Report",
                data=report_resp.content,
                file_name=f"selected_report_{dataset_id}.json",
                mime="application/json",
                use_container_width=True,
                key=f"dl_fs_report_{dataset_id}",
            )
        else:
            st.caption("Report file not available for download yet.")
