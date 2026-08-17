"""Streamlit views for the Phase 2 profiling dashboard (display only)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def _render_name_list(title: str, names: list[str], empty_message: str) -> None:
    if names:
        for name in names:
            st.markdown(f"- `{name}`")
    else:
        st.caption(empty_message)


def render_dataset_preview_table(
    preview: list[dict[str, Any]] | None,
    df: pd.DataFrame | None = None,
    *,
    max_rows: int = 10,
) -> None:
    """Dataset preview for the Profiling page only."""
    st.markdown("##### Dataset Preview — First 10 Rows")
    frame = None
    if preview:
        frame = pd.DataFrame(preview).head(max_rows)
    elif df is not None and not df.empty:
        frame = df.head(max_rows)
    if frame is None or frame.empty:
        st.info("No rows available to preview.")
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_profile_dashboard(
    profile: dict[str, Any],
    *,
    preview: list[dict[str, Any]] | None = None,
    df: pd.DataFrame | None = None,
) -> None:
    """Render the full profiling report using grouped expanders."""
    st.caption("Read-only analysis — your dataset has not been modified.")

    with st.container(border=True):
        render_dataset_preview_table(preview, df, max_rows=10)

    shape = profile["shape"]

    with st.expander("Dataset Shape & Memory", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{shape['rows']:,}")
        c2.metric("Columns", f"{shape['columns']:,}")
        c3.metric("Memory Usage", _format_bytes(profile["memory_usage_bytes"]))

    with st.expander("Column Summary", expanded=True):
        rows = []
        for col in profile["column_summaries"]:
            rows.append(
                {
                    "Name": col["name"],
                    "Data Type": col["data_type"],
                    "Non-null Count": col["non_null_count"],
                    "Null Count": col["null_count"],
                    "Unique Values": col["unique_values"],
                    "Example Value": col["example_value"] or "—",
                    "Inferred Type": col["semantic_type"],
                }
            )
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No columns to summarize.")

    types = profile["column_types"]
    with st.expander("Detected Column Types", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Numerical**")
            _render_name_list("", types["numerical"], "None detected.")
            st.markdown("**Categorical**")
            _render_name_list("", types["categorical"], "None detected.")
            st.markdown("**Datetime**")
            _render_name_list("", types["datetime"], "None detected.")
        with t2:
            st.markdown("**Boolean**")
            _render_name_list("", types["boolean"], "None detected.")
            st.markdown("**Text**")
            _render_name_list("", types["text"], "None detected.")

    with st.expander("Potential Identifier Columns", expanded=False):
        identifiers = profile["identifier_columns"]
        if identifiers:
            id_rows = [
                {
                    "Column": item["column_name"],
                    "Reason": item["reason"],
                    "Recommendation": item["recommendation"],
                }
                for item in identifiers
            ]
            st.dataframe(
                pd.DataFrame(id_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No identifier-like columns detected.")

    with st.expander("Potential Target Columns (Recommendations Only)", expanded=False):
        targets = profile["potential_target_columns"]
        if targets:
            target_rows = [
                {"Column": item["column_name"], "Reason": item["reason"]}
                for item in targets
            ]
            st.dataframe(
                pd.DataFrame(target_rows),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("These columns are suggestions only — nothing was auto-selected.")
        else:
            st.caption("No target-like column names detected.")

    with st.expander("Constant Columns", expanded=False):
        constants = profile["constant_columns"]
        if constants:
            st.warning(
                "These columns have at most one unique value and usually add no "
                "signal for modeling."
            )
            _render_name_list("", constants, "")
        else:
            st.caption("No constant columns detected.")

    with st.expander("High Cardinality Columns", expanded=False):
        high_card = profile["high_cardinality_columns"]
        if high_card:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column Name": item["column_name"],
                            "Unique Count": item["unique_count"],
                            "Recommendation": item["recommendation"],
                        }
                        for item in high_card
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No high-cardinality columns flagged.")
