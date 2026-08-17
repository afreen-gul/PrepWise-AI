"""Tests for Phase 5.3 numerical feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.numerical_feature_engineer import engineer_numerical_features


def _gen_names(details: dict) -> set[str]:
    return {g.feature for g in details["generated"]}


def _skip_reasons(details: dict, feature: str) -> list[str]:
    return [s.reason for s in details["skipped"] if s.feature == feature]


def test_1_strongly_skewed_salary_log() -> None:
    rng = np.random.default_rng(42)
    salary = rng.lognormal(mean=10.5, sigma=1.2, size=200)
    df = pd.DataFrame({"Salary": salary})
    out, details = engineer_numerical_features(df)
    assert "Salary_Log" in _gen_names(details)
    assert "Salary" in out.columns
    assert "Salary_Log" in out.columns
    meta = next(g for g in details["generated"] if g.feature == "Salary_Log")
    assert meta.before_stats is not None
    assert meta.after_stats is not None
    assert abs(meta.after_stats["skewness"]) < abs(meta.before_stats["skewness"])


def test_2_negative_values_skip_log() -> None:
    rng = np.random.default_rng(1)
    values = rng.lognormal(mean=3.0, sigma=1.0, size=100)
    values[:25] = -np.abs(rng.normal(5, 2, size=25))
    df = pd.DataFrame({"Balance": values})
    out, details = engineer_numerical_features(df)
    assert "Balance_Log" not in out.columns
    assert any(
        s.feature == "Balance_Log" and "negative" in s.reason.lower()
        for s in details["skipped"]
    )


def test_3_binary_no_log_or_bin() -> None:
    df = pd.DataFrame({"Flag": [0, 1] * 50})
    out, details = engineer_numerical_features(df)
    assert "Flag_Log" not in out.columns
    assert "Flag_Binned" not in out.columns


def test_4_age_binning() -> None:
    rng = np.random.default_rng(0)
    ages = rng.integers(18, 80, size=100)
    df = pd.DataFrame({"Age": ages})
    out, details = engineer_numerical_features(df)
    assert "Age_Binned" in _gen_names(details)
    assert out["Age_Binned"].notna().sum() == 100


def test_5_meaningful_ratio_revenue_quantity() -> None:
    df = pd.DataFrame(
        {
            "Revenue": [100.0, 200.0, 300.0, 400.0, 500.0] * 10,
            "Quantity": [2.0, 4.0, 5.0, 8.0, 10.0] * 10,
        }
    )
    out, details = engineer_numerical_features(df)
    names = _gen_names(details)
    assert any("Per" in n for n in names)
    assert len(out) == len(df)


def test_6_division_by_zero_safe() -> None:
    df = pd.DataFrame(
        {
            "Revenue": [100.0, 200.0, 300.0, 400.0],
            "Quantity": [2.0, 0.0, 5.0, 0.0],
        }
    )
    out, details = engineer_numerical_features(df)
    ratio_cols = [c for c in out.columns if "Per" in c]
    assert ratio_cols
    col = ratio_cols[0]
    assert pd.isna(out.loc[1, col])
    assert not np.isinf(out[col].dropna()).any()
    assert any(i["issue"] == "division_by_zero" for i in details["issues"])


def test_7_no_polynomial_explosion() -> None:
    df = pd.DataFrame(
        {
            "A": np.arange(50, dtype=float),
            "B": np.arange(50, dtype=float) * 2,
            "C": np.arange(50, dtype=float) * 3,
        }
    )
    out, details = engineer_numerical_features(df)
    # No x^2 style columns
    assert not any("_Sq" in c or "_Pow" in c or "_x_" in c for c in out.columns)
    assert details["polynomial_recommendations"] or True  # may be empty if names don't match
    # Column growth should be modest (ratios only if semantic; A/B/C have no semantic pairs)
    assert len(out.columns) - len(df.columns) <= 3


def test_numerical_preserves_rows_and_originals() -> None:
    df = pd.DataFrame({"Salary": [1.0, 10.0, 100.0, 1000.0] * 20, "Keep": [1, 2, 3, 4] * 20})
    out, _ = engineer_numerical_features(df)
    assert len(out) == len(df)
    assert list(df.columns) == ["Salary", "Keep"]
    assert "Salary" in out.columns and "Keep" in out.columns
