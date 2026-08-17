"""Phase 6.5 — Variance Inflation Factor (multicollinearity)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.schemas.feature_selection import FeatureQualityRow, VIFRow
from app.services.feature_selection_config import (
    MAX_VIF_FEATURES,
    MIN_ROWS_FOR_VIF,
    VIF_REVIEW_THRESHOLD,
    VIF_THRESHOLD,
)


def _safe_vif_values(X: np.ndarray) -> list[float | None]:
    """Compute VIF for each column; return None for unstable columns."""
    n_features = X.shape[1]
    vifs: list[float | None] = []
    for i in range(n_features):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        if others.shape[1] == 0:
            vifs.append(1.0)
            continue
        # Add intercept
        design = np.column_stack([np.ones(len(y)), others])
        try:
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            y_hat = design @ beta
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot <= 1e-12:
                vifs.append(None)
                continue
            r2 = 1.0 - (ss_res / ss_tot)
            r2 = min(max(r2, 0.0), 0.999999)
            vif = 1.0 / (1.0 - r2)
            if not np.isfinite(vif):
                vifs.append(None)
            else:
                vifs.append(float(min(vif, 1e6)))
        except Exception:
            vifs.append(None)
    return vifs


def analyze_vif(
    df: pd.DataFrame,
    *,
    quality_rows: list[FeatureQualityRow],
    target_column: str | None = None,
    correlation_pairs: list[Any] | None = None,
) -> tuple[list[VIFRow], bool, str | None]:
    """Return VIF rows, availability flag, and message when unavailable."""
    quality_by_name = {q.feature: q for q in quality_rows}
    candidates = [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and (target_column is None or str(c) != target_column)
        and not (quality_by_name.get(str(c)) and quality_by_name[str(c)].is_constant)
        and not (quality_by_name.get(str(c)) and quality_by_name[str(c)].is_identifier)
    ]

    if len(df) < MIN_ROWS_FOR_VIF:
        return (
            [],
            False,
            f"VIF unavailable for this dataset because it has fewer than "
            f"{MIN_ROWS_FOR_VIF} rows.",
        )
    if len(candidates) < 2:
        return (
            [],
            False,
            "VIF unavailable for this dataset because fewer than two suitable "
            "numerical predictor features are available.",
        )

    # Prefer features that appear in high-correlation pairs, then fill.
    preferred: list[str] = []
    if correlation_pairs:
        for pair in correlation_pairs:
            for name in (pair.feature_a, pair.feature_b):
                if name in candidates and name not in preferred:
                    preferred.append(name)

    selected = preferred[:MAX_VIF_FEATURES]
    for name in candidates:
        if len(selected) >= MAX_VIF_FEATURES:
            break
        if name not in selected:
            selected.append(name)

    work = df[selected].apply(pd.to_numeric, errors="coerce")
    # Drop rows with any NA among selected columns for stable VIF
    work = work.dropna()
    if len(work) < MIN_ROWS_FOR_VIF:
        return (
            [],
            False,
            "VIF unavailable for this dataset because too many missing values "
            "remain after preparing numerical predictors.",
        )

    # Drop near-zero variance after dropna
    keep_cols: list[str] = []
    for col in work.columns:
        if float(work[col].std(ddof=0)) > 1e-12:
            keep_cols.append(str(col))
    if len(keep_cols) < 2:
        return (
            [],
            False,
            "VIF unavailable for this dataset because numerical features lack "
            "sufficient variation.",
        )
    work = work[keep_cols]

    # Standardize for numerical stability
    values = work.to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds < 1e-12] = 1.0
    X = (values - means) / stds

    try:
        vif_vals = _safe_vif_values(X)
    except Exception:
        return (
            [],
            False,
            "VIF unavailable for this dataset because the numerical matrix "
            "could not be inverted reliably (singular or unstable).",
        )

    # Related features from correlation pairs
    related_map: dict[str, list[str]] = {c: [] for c in keep_cols}
    if correlation_pairs:
        for pair in correlation_pairs:
            a, b = pair.feature_a, pair.feature_b
            if a in related_map and b not in related_map[a]:
                related_map[a].append(b)
            if b in related_map and a not in related_map[b]:
                related_map[b].append(a)

    rows: list[VIFRow] = []
    for col, vif in zip(keep_cols, vif_vals):
        if vif is None:
            status = "UNAVAILABLE"
            rec = "Could not compute a stable VIF for this feature."
            vif_out = None
        elif vif > VIF_THRESHOLD:
            status = "HIGH"
            rec = (
                "High multicollinearity — overlapping information with other "
                "numerical features; review before modeling."
            )
            vif_out = round(vif, 2)
        elif vif >= VIF_REVIEW_THRESHOLD:
            status = "REVIEW"
            rec = "Moderate multicollinearity — review redundancy."
            vif_out = round(vif, 2)
        else:
            status = "GOOD"
            rec = "Generally acceptable multicollinearity."
            vif_out = round(vif, 2)
        rows.append(
            VIFRow(
                feature=col,
                vif=vif_out,
                status=status,
                related_features=related_map.get(col, [])[:8],
                recommendation=rec,
            )
        )

    rows.sort(key=lambda r: (r.vif is None, -(r.vif or 0.0)))
    return rows, True, None
