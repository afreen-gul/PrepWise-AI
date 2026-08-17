"""Phase 6.1 — per-feature quality analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.schemas.feature_selection import FeatureQualityRow
from app.services.duplicate_columns import find_exact_duplicate_column_groups
from app.services.feature_opportunity_detector import detect_identifier
from app.services.feature_selection_config import (
    HIGH_CARDINALITY_THRESHOLD,
    HIGH_MISSING_THRESHOLD,
    MODERATE_MISSING_THRESHOLD,
    NEAR_CONSTANT_THRESHOLD,
)


def _semantic_type(series: pd.Series, name: str, is_id: bool, is_target: bool) -> str:
    if is_target:
        return "target"
    if is_id:
        return "identifier"
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        nunique = int(series.dropna().nunique())
        if nunique <= 2:
            return "binary"
        return "numerical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    nunique = int(series.dropna().nunique())
    n = len(series)
    if n and nunique / n >= HIGH_CARDINALITY_THRESHOLD and nunique > 20:
        # Long text vs high-card categorical
        sample = series.dropna().astype(str).head(30)
        mean_len = float(sample.str.len().mean()) if len(sample) else 0.0
        if mean_len >= 40:
            return "text"
        return "high_cardinality_categorical"
    return "categorical"


def analyze_feature_quality(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
    generated_meta_by_name: dict[str, dict[str, Any]] | None = None,
) -> list[FeatureQualityRow]:
    """Analyze every column for structural quality signals."""
    generated_meta_by_name = generated_meta_by_name or {}
    duplicate_of = find_exact_duplicate_column_groups(df)
    n_rows = len(df)
    rows: list[FeatureQualityRow] = []

    for col in df.columns:
        name = str(col)
        series = df[col]
        non_null = series.dropna()
        missing_pct = float((1.0 - len(non_null) / n_rows) if n_rows else 0.0)
        unique_count = int(non_null.nunique())
        unique_pct = float((unique_count / n_rows) if n_rows else 0.0)

        most_freq = None
        freq_pct = None
        is_constant = False
        is_near_constant = False
        if len(non_null) > 0:
            vc = non_null.astype(str).value_counts(dropna=True)
            most_freq = str(vc.index[0])
            top_share = float(vc.iloc[0] / len(non_null))
            freq_pct = round(top_share * 100.0, 2)
            is_constant = unique_count <= 1
            is_near_constant = (not is_constant) and top_share >= NEAR_CONSTANT_THRESHOLD

        ident = detect_identifier(series, name)
        is_id = bool(ident and ident.get("is_identifier"))
        is_target = target_column is not None and name == target_column
        if is_target:
            is_id = False

        gen = generated_meta_by_name.get(name)
        is_generated = gen is not None
        flags: list[str] = []
        if is_constant:
            flags.append("constant")
        if is_near_constant:
            flags.append("near_constant")
        if missing_pct >= HIGH_MISSING_THRESHOLD:
            flags.append("high_missing")
        elif missing_pct >= MODERATE_MISSING_THRESHOLD:
            flags.append("moderate_missing")
        if is_id:
            flags.append("identifier")
        if name in duplicate_of:
            flags.append("exact_duplicate")
        semantic = _semantic_type(series, name, is_id, is_target)
        if semantic == "high_cardinality_categorical":
            flags.append("high_cardinality")

        dtype = str(series.dtype)
        rows.append(
            FeatureQualityRow(
                feature=name,
                datatype=dtype,
                semantic_type=semantic,
                missing_pct=round(missing_pct * 100.0, 2),
                unique_count=unique_count,
                unique_pct=round(unique_pct * 100.0, 2),
                most_frequent_value=most_freq,
                frequency_pct=freq_pct,
                is_constant=is_constant,
                is_near_constant=is_near_constant,
                is_identifier=is_id,
                is_exact_duplicate=name in duplicate_of,
                duplicate_of=duplicate_of.get(name),
                is_generated=is_generated,
                source_feature=(gen or {}).get("source"),
                transformation=(gen or {}).get("transformation"),
                quality_flags=flags,
            )
        )
    return rows
