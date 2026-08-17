"""Phase 6.6–6.8 — target detection and mutual-information scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.schemas.feature_selection import TargetScoreRow
from app.services.dataset_profiler import _name_matches_target
from app.services.feature_selection_config import MIN_ROWS_FOR_MI, MI_RANDOM_STATE


def detect_target_column(
    df: pd.DataFrame,
    explicit: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Return (target_column, message).

    Prefer explicit target; otherwise use name heuristics only (no invented targets).
    """
    if explicit:
        if explicit not in df.columns:
            return None, (
                f"Requested target column `{explicit}` was not found in the "
                "feature-engineered dataset."
            )
        return explicit, None

    candidates = [str(c) for c in df.columns if _name_matches_target(str(c))]
    if not candidates:
        return None, (
            "No target detected. Target-aware feature selection was skipped."
        )
    # Prefer exact-ish churn/target/label names, else first match
    priority = ("churn", "target", "label", "class", "outcome")
    for key in priority:
        for c in candidates:
            if key in c.lower().replace(" ", "_"):
                return c, None
    return candidates[0], None


def infer_target_task(series: pd.Series) -> str:
    """Return 'classification' or 'regression'."""
    non_null = series.dropna()
    if non_null.empty:
        return "classification"
    if pd.api.types.is_bool_dtype(non_null):
        return "classification"
    nunique = int(non_null.nunique())
    if not pd.api.types.is_numeric_dtype(non_null):
        return "classification"
    # Small discrete numeric → classification
    if nunique <= 20:
        return "classification"
    return "regression"


def _encode_feature_for_mi(series: pd.Series) -> np.ndarray:
    non_null_mask = series.notna()
    values = series.copy()
    if pd.api.types.is_numeric_dtype(values):
        filled = pd.to_numeric(values, errors="coerce")
        median = float(filled.median()) if filled.notna().any() else 0.0
        filled = filled.fillna(median)
        return filled.to_numpy(dtype=float)
    # Categorical / object → codes
    codes = values.astype("category").cat.codes.replace(-1, np.nan)
    mode = codes.mode()
    fill = int(mode.iloc[0]) if len(mode) else 0
    return codes.fillna(fill).to_numpy(dtype=float)


def compute_mutual_information(
    df: pd.DataFrame,
    target_column: str,
    *,
    exclude: set[str] | None = None,
) -> tuple[list[TargetScoreRow], str | None, dict[str, float]]:
    """Lightweight MI scores when a valid target exists."""
    exclude = exclude or set()
    if target_column not in df.columns:
        return [], "Target column missing; skipped mutual information.", {}

    if len(df) < MIN_ROWS_FOR_MI:
        return (
            [],
            f"Mutual information skipped: fewer than {MIN_ROWS_FOR_MI} rows.",
            {},
        )

    y_series = df[target_column]
    if y_series.dropna().nunique() < 2:
        return (
            [],
            "Mutual information skipped: target has insufficient class/value variation.",
            {},
        )

    task = infer_target_task(y_series)
    try:
        from app.services.mutual_info_utils import mutual_info_scores
    except ImportError:
        return (
            [],
            "Mutual information unavailable.",
            {},
        )

    feature_names = [
        str(c)
        for c in df.columns
        if str(c) != target_column and str(c) not in exclude
    ]
    if not feature_names:
        return [], "No predictor features available for mutual information.", {}

    # Align rows where target is present
    mask = y_series.notna()
    if int(mask.sum()) < MIN_ROWS_FOR_MI:
        return (
            [],
            "Mutual information skipped: too many missing target values.",
            {},
        )

    X_cols = []
    discrete_mask: list[bool] = []
    for name in feature_names:
        series = df.loc[mask, name]
        X_cols.append(_encode_feature_for_mi(series))
        discrete_mask.append(not pd.api.types.is_numeric_dtype(df[name]))

    X = np.column_stack(X_cols)
    y_raw = y_series.loc[mask]

    try:
        if task == "classification":
            y_codes, _ = pd.factorize(y_raw.astype(str))
            if len(np.unique(y_codes)) < 2:
                return (
                    [],
                    "Mutual information skipped: target has fewer than two classes.",
                    {},
                )
            scores = mutual_info_scores(
                X,
                y_codes,
                discrete_features=np.array(discrete_mask, dtype=bool),
                task="classification",
                random_state=MI_RANDOM_STATE,
            )
        else:
            y_num = pd.to_numeric(y_raw, errors="coerce")
            valid = y_num.notna()
            if int(valid.sum()) < MIN_ROWS_FOR_MI:
                return (
                    [],
                    "Mutual information skipped: regression target could not be coerced.",
                    {},
                )
            scores = mutual_info_scores(
                X[valid.to_numpy()],
                y_num.loc[valid].to_numpy(dtype=float),
                discrete_features=np.array(discrete_mask, dtype=bool),
                task="regression",
                random_state=MI_RANDOM_STATE,
            )
    except Exception as exc:
        return [], f"Mutual information skipped: {exc}", {}

    scored = list(zip(feature_names, [float(s) for s in scores]))
    scored.sort(key=lambda t: t[1], reverse=True)
    mi_map = {name: score for name, score in scored}

    rows: list[TargetScoreRow] = []
    for rank, (name, score) in enumerate(scored, start=1):
        if score >= 0.1:
            interpretation = "Stronger statistical association with the target."
            rec = "Useful evidence toward KEEP (not absolute)."
        elif score >= 0.02:
            interpretation = "Moderate association with the target."
            rec = "Supportive evidence; combine with other signals."
        else:
            interpretation = (
                "Weak association with the target. A low score does not "
                "automatically mean the feature is useless."
            )
            rec = "REVIEW if other quality issues exist; otherwise KEEP."
        rows.append(
            TargetScoreRow(
                feature=name,
                target_type=task,
                mi_score=round(score, 6),
                rank=rank,
                interpretation=interpretation,
                recommendation=rec,
            )
        )
    return rows, None, mi_map
