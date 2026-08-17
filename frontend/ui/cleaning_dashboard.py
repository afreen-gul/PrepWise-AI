"""Streamlit UI for Phase 4 intelligent data cleaning (display + config only)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st


def build_cleaning_config_from_ui() -> dict[str, Any]:
    """Render the cleaning options panel and return a config dict for the API."""
    with st.container(border=True):
        st.markdown("#### 🧹 Data Cleaning Options")
        st.caption("Configure cleaning before running a dry-run or applying changes.")

        c1, c2 = st.columns(2)
        with c1:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
            remove_dup_cols = st.checkbox(
                "Remove duplicate columns",
                value=True,
                help=(
                    "Drop columns that are exact value copies of an earlier column "
                    "(e.g. City_Copy identical to City)."
                ),
            )
            convert_dtypes = st.checkbox("Convert safe data types", value=True)
            handle_missing = st.checkbox("Handle missing values", value=True)
            handle_invalid = st.checkbox(
                "Handle invalid / domain-violating values",
                value=True,
                help=(
                    "Impossible ages, negative salaries/quantities → missing, "
                    "then imputed if missing-value handling is on."
                ),
            )
        with c2:
            handle_empty = st.checkbox("Treat empty strings as missing", value=True)
            handle_outliers = st.checkbox("Handle outliers (IQR)", value=True)
            remove_constants = st.checkbox("Remove constant columns", value=False)
            drop_high_missing = st.checkbox(
                "Drop high-missingness columns",
                value=False,
                help="If off, high-missing columns are flagged for review only.",
            )

        outlier_strategy = st.radio(
            "Outlier Strategy (statistical IQR outliers only)",
            options=["flag", "remove", "clip"],
            index=0,
            horizontal=True,
            format_func=lambda x: {
                "flag": "Flag",
                "remove": "Remove",
                "clip": "Clip",
            }[x],
            help=(
                "Applies to statistical IQR outliers only. "
                "Domain-invalid values (e.g. impossible ages) are converted to "
                "missing and imputed separately — they are not treated as outliers."
            ),
        )

        threshold_pct = st.slider(
            "High Missingness Threshold (%)",
            min_value=50,
            max_value=100,
            value=70,
            step=5,
        )

    return {
        "remove_duplicate_rows": remove_duplicates,
        "remove_duplicate_columns": remove_dup_cols,
        "convert_safe_dtypes": convert_dtypes,
        "handle_missing_values": handle_missing,
        "remove_constant_columns": remove_constants,
        "drop_high_missing_columns": drop_high_missing,
        "handle_invalid_values": handle_invalid,
        "handle_empty_strings": handle_empty,
        "handle_outliers": handle_outliers,
        "outlier_strategy": outlier_strategy,
        "high_missingness_threshold": threshold_pct / 100.0,
    }


def render_cleaning_summary(summary: dict[str, Any]) -> None:
    """Show issues found / fixed / flagged."""
    with st.container(border=True):
        st.markdown("#### Cleaning Summary")

        st.markdown("**Issues Found**")
        for item in summary.get("issues_found", []):
            st.markdown(f"- {item}")

        fixed = summary.get("issues_to_fix", [])
        st.markdown("**Issues That Will Be Fixed**" if fixed else "**Issues Fixed**")
        if fixed:
            rows = [
                {
                    "Issue": i["issue"],
                    "Before": i["before"],
                    "After": i["after"],
                    "Action": i["action"],
                }
                for i in fixed
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("None.")

        flagged = summary.get("issues_to_flag", [])
        st.markdown("**Issues That Will Be Flagged**" if flagged else "**Issues Flagged**")
        if flagged:
            rows = [
                {
                    "Issue": i["issue"],
                    "Before": i["before"],
                    "After": i["after"],
                    "Action": i["action"],
                }
                for i in flagged
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("None.")


def render_before_after(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Before vs after metric comparison."""
    with st.container(border=True):
        st.markdown("#### Before vs After")
        r1 = st.columns(4)
        r1[0].metric("Rows Before", f"{before['rows']:,}")
        r1[1].metric("Rows After", f"{after['rows']:,}", delta=after["rows"] - before["rows"])
        r1[2].metric("Columns Before", f"{before['columns']:,}")
        r1[3].metric(
            "Columns After",
            f"{after['columns']:,}",
            delta=after["columns"] - before["columns"],
        )

        r2 = st.columns(4)
        r2[0].metric("Missing Before", f"{before['missing_values']:,}")
        r2[1].metric(
            "Missing After",
            f"{after['missing_values']:,}",
            delta=after["missing_values"] - before["missing_values"],
        )
        r2[2].metric("Duplicates Before", f"{before['duplicate_rows']:,}")
        r2[3].metric(
            "Duplicates After",
            f"{after['duplicate_rows']:,}",
            delta=after["duplicate_rows"] - before["duplicate_rows"],
        )

        st.markdown(
            f"**Outliers detected (after):** {after.get('outliers_detected', 0):,}"
        )
        modified = after.get("columns_modified") or []
        if modified:
            st.markdown("**Columns modified:** " + ", ".join(f"`{c}`" for c in modified))
        else:
            st.caption("No columns were structurally modified.")


def render_duplicate_columns_summary(entries: list[dict[str, Any]]) -> None:
    """Duplicate column detection / removal card."""
    detection = next(
        (e for e in entries if e.get("operation") == "duplicate_column_detection"),
        None,
    )
    removals = [
        e for e in entries if e.get("operation") == "duplicate_column_removal"
    ]
    retained = [
        e for e in entries if e.get("operation") == "duplicate_column_retained"
    ]

    rows: list[dict[str, Any]] = []
    if detection:
        details = detection.get("details") or {}
        for item in details.get("duplicate_columns") or []:
            rows.append(
                {
                    "Duplicate Column": item.get("duplicate_column"),
                    "Original Column": item.get("original_column"),
                    "Similarity": f"{item.get('similarity', 100):.0f}%",
                    "Action": "Removed"
                    if any(
                        r.get("column") == item.get("duplicate_column")
                        for r in removals
                    )
                    else "Detected",
                }
            )
    for entry in removals:
        if any(r.get("Duplicate Column") == entry.get("column") for r in rows):
            continue
        d = entry.get("details") or {}
        rows.append(
            {
                "Duplicate Column": entry.get("column"),
                "Original Column": d.get("duplicate_of"),
                "Similarity": f"{d.get('similarity', 100):.0f}%",
                "Action": d.get("action", "Removed"),
            }
        )

    if not rows and not retained:
        return

    with st.container(border=True):
        st.markdown("#### 📋 Duplicate Columns")
        st.caption(f"Detected: **{len(rows) or len(retained)}**")
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        for entry in removals:
            d = entry.get("details") or {}
            st.markdown(
                f"- `{entry.get('column')}` → duplicate of `{d.get('duplicate_of')}` "
                "**[Removed automatically]**"
            )
        for entry in retained:
            d = entry.get("details") or {}
            st.markdown(
                f"- `{entry.get('column')}` → duplicate of `{d.get('duplicate_of')}` "
                "*(retained — option off)*"
            )
        if detection:
            near = (detection.get("details") or {}).get("potentially_redundant") or []
            if near:
                st.markdown("**Potentially redundant** (not removed automatically)")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Column A": p.get("column_a"),
                                "Column B": p.get("column_b"),
                                "Similarity": f"{p.get('similarity')}%",
                            }
                            for p in near
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


def render_categorical_imputation_summary(entries: list[dict[str, Any]]) -> None:
    """Table: Column | Missing | Method | Replacement | Confidence | Reason."""
    summary_rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("operation") != "missing_value_imputation":
            continue
        details = entry.get("details") or {}
        table = details.get("imputation_summary_table")
        if table:
            for row in table:
                conf = row.get("confidence", "—")
                conf_pct = row.get("confidence_pct")
                if conf_pct is not None:
                    conf_display = f"{conf_pct}% ({conf})"
                else:
                    conf_display = str(conf)
                summary_rows.append(
                    {
                        "Column": row.get("column"),
                        "Missing": row.get("missing"),
                        "Method": row.get("method"),
                        "Replacement": row.get("replacement"),
                        "Confidence": conf_display,
                        "Reason": row.get("reason"),
                    }
                )

    if not summary_rows:
        return

    with st.container(border=True):
        st.markdown("#### 🏷 Categorical Imputation Decisions")
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )

        for entry in entries:
            if entry.get("operation") != "missing_value_imputation":
                continue
            details = entry.get("details") or {}
            process = details.get("decision_process")
            if not process:
                table = details.get("imputation_summary_table") or []
                if table:
                    process = table[0].get("decision_process")
            if not process:
                continue
            col_name = entry.get("column") or details.get("column") or "Column"
            with st.expander(
                f"Why did PrepWise choose this method? — {col_name}",
                expanded=False,
            ):
                for step in process:
                    status = step.get("status", "")
                    mark = "✓" if status == "selected" else "○"
                    st.markdown(f"{mark} **{step.get('step', '')}**")
                    st.caption(step.get("detail") or "")
                final_label = details.get("method_label") or details.get("method")
                st.markdown(f"**Final:** {final_label}")


def render_cleaning_log(entries: list[dict[str, Any]]) -> None:
    """Structured cleaning log expander with before/after detail."""
    render_duplicate_columns_summary(entries)
    render_categorical_imputation_summary(entries)
    with st.expander("📜 Cleaning Log", expanded=False):
        if not entries:
            st.caption("No log entries.")
            return

        rows = []
        for entry in entries:
            details = entry.get("details") or {}
            category = details.get("value_category", "")
            before = details.get("before_values")
            if before is None and "before_missing" in details:
                before = f"missing={details.get('before_missing')}"
            after = details.get("after_value", details.get("after_values"))
            if after is None and "after_missing" in details:
                after = f"missing={details.get('after_missing')}"
            rows.append(
                {
                    "Category": category or "—",
                    "Operation": entry.get("operation", ""),
                    "Column": entry.get("column") or "—",
                    "Method": details.get("method_label") or details.get("method") or "—",
                    "Confidence": details.get("confidence") or "—",
                    "Dominant %": (
                        details.get("dominant_percentage")
                        if details.get("dominant_percentage") is not None
                        else "—"
                    ),
                    "Final dtype": details.get("final_dtype") or "—",
                    "Before": _format_log_values(before),
                    "After": _format_log_values(after),
                    "Reason": details.get("reason") or entry.get("message") or "",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Show row-level transformation samples when present
        sample_blocks = [
            e
            for e in entries
            if (e.get("details") or {}).get("transformations")
        ]
        if sample_blocks:
            st.markdown("**Transformation samples (before → after)**")
            for entry in sample_blocks:
                details = entry.get("details") or {}
                transforms = details.get("transformations") or []
                label = entry.get("column") or entry.get("operation")
                category = details.get("value_category", "")
                st.caption(
                    f"{label} · {entry.get('operation')}"
                    + (f" · {category}" if category else "")
                )
                st.dataframe(
                    pd.DataFrame(transforms),
                    use_container_width=True,
                    hide_index=True,
                )


def _format_log_values(value: Any) -> str:
    if value is None or value == "" or value == []:
        return "—"
    if isinstance(value, list):
        preview = ", ".join("missing" if v is None else str(v) for v in value[:6])
        if len(value) > 6:
            preview += ", …"
        return preview
    return str(value)


def render_cleaned_preview(preview: list[dict[str, Any]]) -> None:
    with st.container(border=True):
        st.markdown("#### 📄 Cleaned Dataset Preview")
        if preview:
            st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
        else:
            st.info("Cleaned dataset has no rows to preview.")


def render_download_button(
    backend_url: str,
    dataset_id: int,
    cleaned_filename: str,
) -> None:
    """Fetch cleaned CSV bytes and offer a Streamlit download button."""
    url = f"{backend_url}/api/v1/datasets/{dataset_id}/clean/download"
    try:
        response = requests.get(url, timeout=60)
    except requests.exceptions.RequestException:
        st.error("Could not download cleaned dataset from the backend.")
        return

    if response.status_code != 200:
        st.error(f"Download failed: {response.text}")
        return

    st.download_button(
        label="⬇ Download Cleaned Dataset",
        data=response.content,
        file_name=cleaned_filename or "cleaned_dataset.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
