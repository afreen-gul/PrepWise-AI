"""Streamlit views for Phase 3 data quality reports (display only)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def _severity_badge(severity: str) -> str:
    mapping = {
        "Low": "🟢 Low",
        "Medium": "🟡 Medium",
        "High": "🔴 High",
    }
    return mapping.get(severity, severity)


def _score_style(level: str) -> str:
    styles = {
        "Excellent": ("🟢", "success"),
        "Good": ("🔵", "info"),
        "Fair": ("🟡", "warning"),
        "Poor": ("🔴", "error"),
    }
    return styles.get(level, ("⚪", "info"))


def render_quality_score_card(report: dict[str, Any]) -> None:
    """Top-level quality score banner."""
    score_info = report["quality_score"]
    icon, _ = _score_style(score_info["level"])

    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Quality Score", f"{score_info['score']} / 100")
        with c2:
            st.markdown(f"### {icon} Quality Level: **{score_info['level']}**")
            st.caption(
                "Excellent (90–100) · Good (75–89) · Fair (60–74) · Poor (<60)"
            )


def render_quality_dashboard(report: dict[str, Any]) -> None:
    """Render the data quality report with cards and expanders."""
    st.caption("Read-only assessment — no changes were made to your dataset.")

    render_quality_score_card(report)

    missing = report.get("missing_values") or []
    outliers = report.get("outliers") or []
    dup_rows = report.get("duplicate_rows") or {}
    constants = report.get("constant_columns") or []
    dup_cols = report.get("duplicate_columns") or []

    st.markdown("##### Issues Detected")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Missing Values", f"{len(missing)} columns")
    m2.metric("Potential Outliers", f"{len(outliers)} columns")
    m3.metric("Duplicate Rows", f"{dup_rows.get('count', 0):,}")
    m4.metric("Constant Columns", f"{len(constants)}")
    m5.metric("Duplicate Columns", f"{len(dup_cols)}")
    st.markdown("")

    breakdown = report["score_breakdown"]
    with st.expander("Score Breakdown (points deducted)", expanded=False):
        rows = [
            {"Issue area": "Missing values", "Points deducted": breakdown["missing_values"]},
            {"Issue area": "Duplicate rows", "Points deducted": breakdown["duplicate_rows"]},
            {"Issue area": "Constant columns", "Points deducted": breakdown["constant_columns"]},
            {
                "Issue area": "High missing columns (>50%)",
                "Points deducted": breakdown["high_missing_columns"],
            },
            {"Issue area": "Invalid values", "Points deducted": breakdown["invalid_values"]},
            {"Issue area": "Outliers (IQR)", "Points deducted": breakdown["outliers"]},
            {"Issue area": "Class imbalance", "Points deducted": breakdown["class_imbalance"]},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Missing Values", expanded=True):
        if missing:
            rows = [
                {
                    "Column": m["column_name"],
                    "Count": m["count"],
                    "Percentage": f"{m['percentage']:.2f}%",
                    "Severity": _severity_badge(m["severity"]),
                }
                for m in missing
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.success("No missing values detected.")

    with st.expander("Duplicate Rows", expanded=False):
        with st.container(border=True):
            st.metric("Duplicate row count", f"{dup_rows.get('count', 0):,}")
            st.caption(
                f"{dup_rows.get('percentage', 0):.2f}% of all rows are exact duplicates."
            )

    dup_cols = report["duplicate_columns"]
    near_cols = report.get("potentially_redundant_columns") or []
    with st.expander("📋 Duplicate Columns", expanded=False):
        if dup_cols:
            st.markdown("**Duplicate Columns Detected**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Duplicate Column": p.get("duplicate_column") or p["column_b"],
                            "Original Column": p.get("original_column") or p["column_a"],
                            "Similarity": f"{p.get('similarity', 100):.0f}%",
                            "Action": "Would remove on clean",
                        }
                        for p in dup_cols
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No exact duplicate columns detected.")
        if near_cols:
            st.markdown("**Potentially redundant** (not removed automatically)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column A": p["column_a"],
                            "Column B": p["column_b"],
                            "Similarity": f"{p['similarity']}%",
                        }
                        for p in near_cols
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("➖ Constant Columns", expanded=False):
        constants = report["constant_columns"]
        if constants:
            st.warning("These columns contain only one unique value.")
            for name in constants:
                st.markdown(f"- `{name}`")
        else:
            st.success("No constant columns detected.")

    with st.expander("⚠ High Missing Columns (>50%)", expanded=False):
        high_missing = report["high_missing_columns"]
        if high_missing:
            for name in high_missing:
                st.markdown(f"- 🔴 `{name}`")
        else:
            st.success("No columns exceed 50% missing values.")

    with st.expander("🧪 Suspicious Data Types", expanded=False):
        suspicious = report["suspicious_data_types"]
        if suspicious:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column": s["column_name"],
                            "Issue": s["issue_type"].replace("_", " ").title(),
                            "Details": s["description"],
                        }
                        for s in suspicious
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No suspicious data types detected.")

    with st.expander("❌ Invalid Values", expanded=False):
        invalid = report["invalid_values"]
        if invalid:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column": i["column_name"],
                            "Type": i["issue_type"].replace("_", " ").title(),
                            "Count": i["count"],
                            "Description": i["description"],
                        }
                        for i in invalid
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No invalid values detected by heuristic rules.")

    with st.expander("📊 Outliers (IQR — report only)", expanded=False):
        outliers = report["outliers"]
        if outliers:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column": o["column_name"],
                            "Outlier count": o["outlier_count"],
                            "Lower bound": o["lower_bound"],
                            "Upper bound": o["upper_bound"],
                        }
                        for o in outliers
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Outliers are reported only and were not removed.")
        else:
            st.success("No IQR outliers flagged on numeric columns.")

    imbalance = report["class_imbalance"]
    with st.expander("⚖ Class Imbalance", expanded=False):
        st.markdown(f"**Target column:** `{imbalance['target_column'] or '—'}`")
        st.caption(imbalance["message"])
        if imbalance["distribution"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Class": d["label"],
                            "Count": d["count"],
                            "Percentage": f"{d['percentage']:.2f}%",
                        }
                        for d in imbalance["distribution"]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            if imbalance["is_severe"]:
                st.error("Severe imbalance detected.")
            elif imbalance["majority_percentage"] and imbalance["majority_percentage"] >= 80:
                st.warning("Moderate imbalance detected.")
