"""Tests for duplicate column detection and removal (Phase 4)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.schemas.cleaning import CleaningConfig
from app.services.data_cleaner import run_cleaning_pipeline
from app.services.duplicate_columns import (
    columns_are_exact_duplicates,
    find_exact_duplicate_column_groups,
    list_exact_duplicate_pairs,
    process_duplicate_columns,
)


def test_1_exact_duplicate_removed() -> None:
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]})
    log: list = []
    out = process_duplicate_columns(df, log, remove=True)
    assert list(out.columns) == ["A"]
    assert any(e.operation == "duplicate_column_removal" for e in log)
    entry = next(e for e in log if e.operation == "duplicate_column_removal")
    assert entry.details["duplicate_of"] == "A"
    assert entry.details["similarity"] == 100.0


def test_2_duplicate_with_aligned_missing_values() -> None:
    df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [1, 2, np.nan, 4]})
    assert columns_are_exact_duplicates(df["A"], df["B"])
    log: list = []
    out = process_duplicate_columns(df, log, remove=True)
    assert list(out.columns) == ["A"]


def test_3_similar_not_identical_not_removed() -> None:
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 2, 3, 5]})
    assert not columns_are_exact_duplicates(df["A"], df["B"])
    log: list = []
    out = process_duplicate_columns(df, log, remove=True)
    assert list(out.columns) == ["A", "B"]
    assert not any(e.operation == "duplicate_column_removal" for e in log)


def test_4_misaligned_missing_not_duplicates() -> None:
    df = pd.DataFrame(
        {"A": ["Lahore", "Karachi", np.nan, "Islamabad"], "B": ["Lahore", "Karachi", "Lahore", "Islamabad"]}
    )
    assert not columns_are_exact_duplicates(df["A"], df["B"])


def test_5_different_names_same_values() -> None:
    df = pd.DataFrame({"City": ["a", "b"], "City_Code": ["x", "y"]})
    assert find_exact_duplicate_column_groups(df) == {}


def test_6_multiple_duplicates_keep_first_in_order() -> None:
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3]})
    dup = find_exact_duplicate_column_groups(df)
    assert dup == {"B": "A", "C": "A"}
    pairs = list_exact_duplicate_pairs(df)
    assert len(pairs) == 2
    log: list = []
    out = process_duplicate_columns(df, log, remove=True)
    assert list(out.columns) == ["A"]


def test_7_large_dataset_performance() -> None:
    n = 5000
    rng = np.random.default_rng(0)
    base = rng.integers(0, 100, size=n)
    df = pd.DataFrame(
        {
            "A": base,
            "B": base.copy(),
            "C": rng.integers(0, 100, size=n),
        }
    )
    start = time.perf_counter()
    dup = find_exact_duplicate_column_groups(df)
    elapsed = time.perf_counter() - start
    assert dup.get("B") == "A"
    assert "C" not in dup
    assert elapsed < 5.0


def test_pipeline_stress_city_copy_removed(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "uploads" / "0f28ff04_prepwise_stress_test.csv"
    if not path.exists():
        paths = list(Path(__file__).resolve().parents[1].glob("uploads/*stress*.csv"))
        assert paths, "stress test CSV not found"
        path = paths[0]
    original_bytes = path.read_bytes()
    df = pd.read_csv(path)
    assert "City" in df.columns and "City_Copy" in df.columns
    config = CleaningConfig(
        remove_duplicate_columns=True,
        remove_duplicate_rows=False,
        handle_missing_values=False,
        handle_invalid_values=False,
        handle_empty_strings=False,
        handle_outliers=False,
        convert_safe_dtypes=False,
    )
    cleaned, log, _ = run_cleaning_pipeline(df, config)
    assert path.read_bytes() == original_bytes
    assert "City" in cleaned.columns
    assert "City_Copy" not in cleaned.columns
    assert any(
        e.operation == "duplicate_column_removal" and e.column == "City_Copy"
        for e in log
    )
