"""Streamlit UI for Phase 5 feature-engineering (opportunity detection + datetime FE)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st


def render_feature_candidate_selector(report: dict[str, Any]) -> list[str]:
    """Show recommended candidates with multiselect; return selected ids."""
    candidates = report.get("candidates") or []
    with st.container(border=True):
        st.markdown("##### Feature Engineering Recommendations")
        st.caption(report.get("message") or "")
        st.caption(
            f"Source: **{report.get('source', 'cleaned')}** · "
            f"Candidates: **{report.get('candidates_count', 0)}** · "
            "Nothing has been created yet."
        )

        if not candidates:
            st.info(
                "No engineered features are recommended. "
                "You can still continue without engineering — cleaned columns "
                "will be passed through for Phase 6."
            )
            return []

        rows = [
            {
                "Feature": c.get("feature"),
                "Source": c.get("source"),
                "Category": c.get("category"),
                "Transformation": c.get("transformation"),
                "Priority": c.get("priority"),
                "Reason": c.get("reason"),
            }
            for c in candidates
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        options = [c.get("id") or c.get("feature") for c in candidates]
        options = [o for o in options if o]
        defaults = [
            c.get("id") or c.get("feature")
            for c in candidates
            if c.get("default_selected", True)
        ]
        defaults = [d for d in defaults if d in options]

        selected = st.multiselect(
            "Select engineered features to create",
            options=options,
            default=defaults,
            key="fe_candidate_multiselect",
            help=(
                "Only checked features will be generated. "
                "Leave empty to pass cleaned data through with no new features."
            ),
        )
        st.caption(
            f"{len(selected)} of {len(options)} candidate(s) selected. "
            "Original cleaned columns are always kept."
        )
        return list(selected)


def render_feature_opportunity_dashboard(report: dict[str, Any]) -> None:
    """Display feature types and recommended opportunities (no transforms)."""
    with st.container(border=True):
        st.markdown("#### ⚙ Feature Engineering Opportunities")
        st.caption(
            report.get("message")
            or "Analysis only — no feature transformations were applied."
        )
        st.caption(
            f"Source analyzed: **{report.get('source', '—')}** · "
            f"Transformations applied: "
            f"**{'Yes' if report.get('transformations_applied') else 'No'}**"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Features analyzed", f"{report.get('columns_analyzed', 0):,}")
        m2.metric(
            "Potential opportunities",
            f"{report.get('opportunities_detected', 0):,}",
        )
        m3.metric(
            "Column count unchanged",
            "Yes" if report.get("column_count_unchanged", True) else "No",
        )

        targets = report.get("potential_targets") or []
        if targets:
            st.info(
                "Potential target / leakage-sensitive columns: "
                + ", ".join(f"`{t}`" for t in targets)
            )

        analyses = report.get("column_analyses") or []
        if analyses:
            rows = [
                {
                    "Column": a.get("column"),
                    "Type": a.get("detected_type"),
                    "Opportunity": a.get("opportunity"),
                    "Priority": a.get("priority"),
                    "Reason": a.get("reason"),
                }
                for a in analyses
            ]
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        relationships = report.get("relationships") or []
        if relationships:
            st.markdown("**Suggested column relationships** (not created)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Columns": ", ".join(r.get("columns") or []),
                            "Opportunity": r.get("opportunity"),
                            "Priority": r.get("priority"),
                            "Reason": r.get("reason"),
                        }
                        for r in relationships
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Feature Analysis Details", expanded=False):
            if not analyses:
                st.caption("No column analyses available.")
                return
            for a in analyses:
                st.markdown(
                    f"**`{a.get('column')}`** — {a.get('detected_type')} "
                    f"({a.get('priority')})"
                )
                st.caption(a.get("characteristics") or "")
                st.caption(a.get("reason") or "")
                details = a.get("details") or {}
                if details:
                    st.json(details)
                st.divider()


def render_datetime_feature_engineering_result(result: dict[str, Any]) -> None:
    """Phase 5.2 results: generated / skipped datetime features."""
    with st.container(border=True):
        st.markdown("#### 📅 Feature Engineering Summary (Datetime)")
        st.caption(result.get("message") or "")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Datetime columns analyzed",
            f"{result.get('datetime_columns_analyzed', 0):,}",
        )
        m2.metric("Features generated", f"{result.get('features_generated', 0):,}")
        m3.metric("Features skipped", f"{result.get('features_skipped', 0):,}")

        before = result.get("before") or {}
        after = result.get("after") or {}
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Rows before", f"{before.get('rows', 0):,}")
        b2.metric(
            "Rows after",
            f"{after.get('rows', 0):,}",
            delta=after.get("rows", 0) - before.get("rows", 0),
        )
        b3.metric("Columns before", f"{before.get('columns', 0):,}")
        b4.metric(
            "Columns after",
            f"{after.get('columns', 0):,}",
            delta=after.get("columns", 0) - before.get("columns", 0),
        )
        st.caption(
            f"New features: **{result.get('new_features', 0)}** · "
            f"Original datetime columns preserved: "
            f"**{'Yes' if result.get('original_datetime_columns_preserved') else 'No'}** · "
            f"Source: **{result.get('source', '—')}**"
        )

        analyses = result.get("datetime_analyses") or []
        if analyses:
            with st.expander("Datetime column analysis", expanded=False):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Column": a.get("column"),
                                "Min": a.get("min_date"),
                                "Max": a.get("max_date"),
                                "Unique dates": a.get("unique_dates"),
                                "Range (days)": a.get("date_range_days"),
                                "Contains time": a.get("contains_time"),
                                "Invalid": a.get("invalid_count"),
                            }
                            for a in analyses
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        generated = result.get("generated_features") or []
        st.markdown("**Generated Features**")
        if generated:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "New Feature": g.get("feature"),
                            "Source": g.get("source"),
                            "Transformation": g.get("transformation"),
                            "Reason": g.get("reason"),
                        }
                        for g in generated
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No datetime features were generated.")

        skipped = result.get("skipped_features") or []
        st.markdown("**Skipped Features**")
        if skipped:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Feature": s.get("feature"),
                            "Source": s.get("source"),
                            "Reason": s.get("reason"),
                        }
                        for s in skipped
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No features were skipped.")

        issues = result.get("issues") or []
        if issues:
            st.markdown("**Datetime issues flagged**")
            for issue in issues:
                st.warning(
                    f"{issue.get('issue')}: {issue.get('message')} "
                    f"(columns: {', '.join(issue.get('columns') or [])})"
                )

        preview = result.get("preview") or []
        if preview:
            with st.expander("Featured preview (first rows)", expanded=False):
                st.dataframe(pd.DataFrame(preview), use_container_width=True)


def render_phase5_feature_engineering_result(result: dict[str, Any]) -> None:
    """Full Phase 5 summary: datetime + numerical + text + validation."""
    with st.container(border=True):
        st.markdown("#### FEATURE ENGINEERING SUMMARY")
        st.caption(result.get("message") or "")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Datetime features", f"{result.get('datetime_features_generated', 0):,}")
        m2.metric("Numerical features", f"{result.get('numerical_features_generated', 0):,}")
        m3.metric("Text features", f"{result.get('text_features_generated', 0):,}")
        m4.metric("Final feature count", f"{result.get('final_feature_count', 0):,}")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Generated", f"{result.get('features_generated', 0):,}")
        s2.metric("Skipped", f"{result.get('features_skipped', 0):,}")
        s3.metric("Removed (validation)", f"{result.get('features_removed', 0):,}")
        s4.metric(
            "Original count",
            f"{result.get('original_feature_count', 0):,}",
        )

        before = result.get("before") or {}
        after = result.get("after") or {}
        st.caption(
            f"Rows: {before.get('rows', 0):,} → {after.get('rows', 0):,} · "
            f"Columns: {before.get('columns', 0):,} → {after.get('columns', 0):,} · "
            f"Source: **{result.get('source', '—')}** · "
            f"Originals preserved: "
            f"**{'Yes' if result.get('original_columns_preserved') else 'No'}**"
        )

        generated = result.get("generated_features") or []
        st.markdown("**GENERATED FEATURES**")
        if generated:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Feature": g.get("feature"),
                            "Source": g.get("source"),
                            "Type": g.get("category") or g.get("feature_type"),
                            "Transformation": g.get("transformation"),
                            "Status": g.get("status", "Created"),
                            "Reason": g.get("reason"),
                        }
                        for g in generated
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No features kept after validation.")

        skipped = result.get("skipped_features") or []
        st.markdown("**SKIPPED FEATURES**")
        if skipped:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Feature": s.get("feature"),
                            "Source": s.get("source"),
                            "Reason": s.get("reason"),
                        }
                        for s in skipped
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("None.")

        removed = result.get("removed_features") or []
        st.markdown("**REMOVED FEATURES**")
        if removed:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Feature": r.get("feature"),
                            "Reason": r.get("reason"),
                        }
                        for r in removed
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("None removed during validation.")

        polys = result.get("polynomial_recommendations") or []
        if polys:
            with st.expander("Polynomial recommendations (not applied)", expanded=False):
                for p in polys:
                    st.markdown(f"- {p}")

        issues = result.get("issues") or []
        if issues:
            st.markdown("**Issues flagged**")
            for issue in issues:
                st.warning(
                    f"{issue.get('issue')}: {issue.get('message')}"
                )

        preview = result.get("preview") or []
        if preview:
            with st.expander("Featured preview (first rows)", expanded=False):
                st.dataframe(pd.DataFrame(preview), use_container_width=True)


def render_featured_download_button(
    backend_url: str,
    dataset_id: int,
    filename: str,
    *,
    full_phase5: bool = False,
) -> None:
    if full_phase5:
        url = f"{backend_url}/api/v1/datasets/{dataset_id}/feature-engineering/download"
    else:
        url = (
            f"{backend_url}/api/v1/datasets/{dataset_id}/"
            "feature-engineering/datetime/download"
        )
    try:
        response = requests.get(url, timeout=60)
    except requests.exceptions.RequestException as exc:
        st.error(f"Download failed: {exc}")
        return
    if response.status_code != 200:
        st.error("Featured file not available yet.")
        return
    st.download_button(
        label=f"⬇ Download {filename}",
        data=response.content,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
        key=f"dl_{'phase5' if full_phase5 else 'dt'}_{dataset_id}_{filename}",
    )
