"""Tests for Phase 5.4 text FE, 5.5 validation, and full Phase 5 pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.feature_engineering_pipeline import run_feature_engineering_pipeline
from app.services.feature_validator import validate_and_finalize_features
from app.services.numerical_feature_engineer import engineer_numerical_features
from app.services.text_feature_engineer import engineer_text_features
from app.schemas.feature_engineering import GeneratedFeatureMeta
from app.services.datetime_feature_engineer import engineer_datetime_features
from app.services.feature_opportunity_detector import (
    detect_feature_type,
    generate_feature_engineering_opportunities,
)


def test_8_text_review_features() -> None:
    reviews = [
        "I really enjoyed this product and would buy again.",
        "Terrible quality, not worth the money at all.",
        "Average experience overall with shipping delays.",
        "Excellent packaging and fast delivery service today.",
        "The description did not match the item received.",
    ] * 5
    df = pd.DataFrame({"Review": reviews})
    out, details = engineer_text_features(df)
    names = {g.feature for g in details["generated"]}
    assert "Review_CharCount" in names
    assert "Review_WordCount" in names
    assert out.loc[0, "Review_WordCount"] > 0


def test_9_empty_text_no_crash() -> None:
    df = pd.DataFrame(
        {
            "Review": [
                "Good product with solid build quality overall.",
                None,
                "",
                "   ",
                "Another decent review with enough words here.",
            ]
            * 5
        }
    )
    out, details = engineer_text_features(df)
    assert len(out) == len(df)
    if "Review_WordCount" in out.columns:
        assert int(out.loc[2, "Review_WordCount"]) == 0
        assert int(out.loc[3, "Review_WordCount"]) == 0


def test_10_identifier_no_text_or_numeric_fe() -> None:
    df = pd.DataFrame(
        {
            "Customer_ID": [f"C{i:04d}" for i in range(50)],
            "Value": list(range(50)),
        }
    )
    out_t, det_t = engineer_text_features(df)
    assert "Customer_ID_WordCount" not in out_t.columns
    out_n, det_n = engineer_numerical_features(df)
    assert "Customer_ID_Log" not in out_n.columns


def test_11_duplicate_generated_removed_original_kept() -> None:
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5] * 10})
    working = df.copy()
    working["A_Copy_Gen"] = working["A"]
    meta = [
        GeneratedFeatureMeta(
            feature="A_Copy_Gen",
            source="A",
            feature_type="Integer",
            category="numerical",
            transformation="copy",
            reason="test",
            rows_affected=50,
        )
    ]
    out, details = validate_and_finalize_features(
        working,
        original_columns=["A"],
        generated_meta=meta,
        expected_rows=50,
    )
    assert "A" in out.columns
    assert "A_Copy_Gen" not in out.columns
    assert any("duplicate" in r.reason.lower() for r in details["removed"])


def test_12_constant_generated_removed() -> None:
    df = pd.DataFrame({"A": [1, 2, 3]})
    working = df.copy()
    working["Const_Gen"] = 5
    meta = [
        GeneratedFeatureMeta(
            feature="Const_Gen",
            source="A",
            feature_type="Integer",
            category="numerical",
            transformation="const",
            reason="test",
            rows_affected=3,
        )
    ]
    out, details = validate_and_finalize_features(
        working,
        original_columns=["A"],
        generated_meta=meta,
        expected_rows=3,
    )
    assert "Const_Gen" not in out.columns
    assert any("constant" in r.reason.lower() for r in details["removed"])


def test_13_infinite_handled() -> None:
    df = pd.DataFrame({"A": [1.0, 2.0]})
    working = df.copy()
    working["Bad_Ratio"] = [1.0, np.inf]
    meta = [
        GeneratedFeatureMeta(
            feature="Bad_Ratio",
            source="A",
            feature_type="Float",
            category="numerical",
            transformation="ratio",
            reason="test",
            rows_affected=2,
        )
    ]
    out, details = validate_and_finalize_features(
        working,
        original_columns=["A"],
        generated_meta=meta,
        expected_rows=2,
    )
    if "Bad_Ratio" in out.columns:
        assert not np.isinf(out["Bad_Ratio"].dropna()).any()
    assert any(i.issue == "infinite_values" for i in details["issues"])


def test_14_datetime_regression() -> None:
    df = pd.DataFrame(
        {"Join_Date": ["2024-01-15", "2024-02-20", "2025-03-10"]}
    )
    out, details = engineer_datetime_features(df)
    assert "Join_Date_Year" in out.columns
    assert "Join_Date" in out.columns


def test_15_feature_type_detection_regression() -> None:
    df = pd.DataFrame(
        {
            "Age": [20, 30, 40],
            "Gender": ["M", "F", "M"],
            "Customer_ID": ["a", "b", "c"],
            "Join_Date": ["2024-01-01", "2024-02-01", "2024-03-01"],
        }
    )
    analyses, _, _ = generate_feature_engineering_opportunities(df)
    by = {a.column: a.detected_type for a in analyses}
    assert by["Age"] == "numerical"
    assert by["Gender"] == "categorical"
    assert by["Customer_ID"] == "identifier"
    assert by["Join_Date"] == "datetime"


def test_17_18_pipeline_rows_and_originals() -> None:
    df = pd.DataFrame(
        {
            "Join_Date": pd.date_range("2020-01-01", periods=100, freq="D").astype(str),
            "Salary": np.random.default_rng(0).lognormal(10, 1, 100),
            "Age": np.random.default_rng(1).integers(18, 70, 100),
            "Review": [
                "This is a sufficiently long review text for testing."
            ]
            * 100,
            "Customer_ID": [f"id_{i}" for i in range(100)],
        }
    )
    originals = list(df.columns)
    out, details = run_feature_engineering_pipeline(df)
    assert len(out) == 100
    for col in originals:
        assert col in out.columns
    assert details["generated"] or details["skipped"]


def test_19_large_pipeline() -> None:
    n = 3000
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "Join_Date": pd.date_range("2018-01-01", periods=n, freq="h").astype(str),
            "Salary": rng.lognormal(10, 1.1, n),
            "Age": rng.integers(18, 80, n),
            "Review": ["A reasonably long product review text."] * n,
        }
    )
    start = time.perf_counter()
    out, details = run_feature_engineering_pipeline(df)
    elapsed = time.perf_counter() - start
    assert len(out) == n
    assert elapsed < 30.0
    assert "Join_Date" in out.columns


def test_stress_full_phase5_pipeline() -> None:
    path = Path(__file__).resolve().parents[1] / "uploads" / "0f28ff04_prepwise_stress_test.csv"
    if not path.exists():
        paths = list(Path(__file__).resolve().parents[1].glob("uploads/*stress*.csv"))
        assert paths
        path = paths[0]
    original_bytes = path.read_bytes()
    df = pd.read_csv(path)
    before_cols = list(df.columns)
    before_rows = len(df)
    out, details = run_feature_engineering_pipeline(df)
    assert path.read_bytes() == original_bytes
    assert list(df.columns) == before_cols
    assert len(out) == before_rows
    for col in before_cols:
        assert col in out.columns
    # No constant kept generated features
    for g in details["generated"]:
        assert g.feature in out.columns
        assert out[g.feature].nunique(dropna=True) > 1
