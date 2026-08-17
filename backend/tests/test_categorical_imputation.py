"""Tests for hierarchical categorical missing-value imputation (Phase 4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.schemas.cleaning import CleaningConfig, OutlierStrategy
from app.services.data_cleaner import (
    CategoricalImputationParams,
    find_grouping_candidates,
    impute_missing_values,
    run_cleaning_pipeline,
)
from app.services.duplicate_columns import find_exact_duplicate_column_groups

# Relaxed thresholds for small fixtures; production defaults remain in CleaningConfig.
TEST_CAT_PARAMS = CategoricalImputationParams(
    min_group_size=2,
    min_valid_observations=3,
)


def _impute(
    df: pd.DataFrame,
    params: CategoricalImputationParams | None = None,
) -> tuple[pd.DataFrame, list]:
    log: list = []
    cleaned = impute_missing_values(
        df,
        log,
        high_missing_threshold=0.7,
        drop_high_missing=False,
        categorical_params=params or TEST_CAT_PARAMS,
    )
    return cleaned, log


def test_1_strong_global_mode() -> None:
    """Male = 80%, Female = 20% → Global Mode."""
    genders = ["Male"] * 80 + ["Female"] * 20 + [None]
    df = pd.DataFrame({"Gender": genders})
    cleaned, log = _impute(df)
    entry = next(e for e in log if e.operation == "missing_value_imputation")
    assert entry.details["method"] == "global_mode"
    assert cleaned["Gender"].iloc[-1] == "Male"


def test_2_balanced_categories_distribution() -> None:
    """Male = 51%, Female = 49% → Distribution-Based Imputation."""
    genders = ["Male"] * 51 + ["Female"] * 49 + [None] * 10
    df = pd.DataFrame({"Gender": genders})
    cleaned, log = _impute(df)
    entry = next(e for e in log if e.operation == "missing_value_imputation")
    assert entry.details["method"] == "distribution_based"
    assert cleaned["Gender"].iloc[-10:].notna().all()
    assert set(cleaned["Gender"].iloc[-10:].unique()) <= {"Male", "Female"}


def test_3_multiple_categories_distribution() -> None:
    """Near-uniform cities → Distribution-Based (not global Lahore)."""
    cities = (
        ["Lahore"] * 21
        + ["Karachi"] * 20
        + ["Islamabad"] * 20
        + ["Multan"] * 19
        + ["Peshawar"] * 20
        + [None] * 5
    )
    df = pd.DataFrame({"City": cities})
    cleaned, log = _impute(df)
    entry = next(e for e in log if e.operation == "missing_value_imputation")
    assert entry.details["method"] == "distribution_based"
    filled = cleaned.loc[df["City"].isna(), "City"]
    assert len(filled) == 5
    assert not (filled == "Lahore").all()


def test_4_insufficient_observations_unknown() -> None:
    """Too few valid observations for global/distribution → Unknown."""
    df = pd.DataFrame({"Category": ["A"] * 5 + ["B"] * 5 + [None] * 5})
    params = CategoricalImputationParams(min_group_size=2, min_valid_observations=20)
    cleaned, log = _impute(df, params=params)
    entry = next(e for e in log if e.operation == "missing_value_imputation")
    assert entry.details["method"] == "unknown_category"
    assert (cleaned["Category"] == "Unknown").sum() == 5


def test_5_group_relationship_priority() -> None:
    rows: list[dict[str, str | None]] = []
    for _ in range(12):
        rows.append({"City": "Lahore", "Gender": "Male"})
    rows.append({"City": "Lahore", "Gender": "Female"})
    rows.append({"City": "Lahore", "Gender": None})

    for _ in range(12):
        rows.append({"City": "Karachi", "Gender": "Female"})
    rows.append({"City": "Karachi", "Gender": "Male"})
    rows.append({"City": "Karachi", "Gender": None})

    df = pd.DataFrame(rows)
    cleaned, log = _impute(df)
    entry = next(e for e in log if e.column == "Gender")
    lahore_missing_idx = df.index[(df["City"] == "Lahore") & df["Gender"].isna()]
    karachi_missing_idx = df.index[(df["City"] == "Karachi") & df["Gender"].isna()]
    assert cleaned.loc[lahore_missing_idx, "Gender"].iloc[0] == "Male"
    assert cleaned.loc[karachi_missing_idx, "Gender"].iloc[0] == "Female"
    assert entry.details["method"] == "group_based_mode"


def test_6_customer_id_not_grouping_feature() -> None:
    df = pd.DataFrame(
        {
            "Customer_ID": [f"id_{i}" for i in range(20)],
            "Gender": ["Male"] * 19 + [None],
        }
    )
    candidates = find_grouping_candidates(df, "Gender", TEST_CAT_PARAMS)
    assert "Customer_ID" not in candidates


def test_7_duplicate_column_excluded_from_grouping() -> None:
    n = 15
    df = pd.DataFrame(
        {
            "City": ["Lahore"] * n + ["Karachi"] * n,
            "City_Copy": ["Lahore"] * n + ["Karachi"] * n,
            "Gender": ["Male"] * (2 * n - 1) + [None],
        }
    )
    dup = find_exact_duplicate_column_groups(df)
    assert dup.get("City_Copy") == "City"
    candidates = find_grouping_candidates(
        df, "Gender", TEST_CAT_PARAMS, duplicate_of=dup
    )
    assert "City_Copy" not in candidates
    assert "City" in candidates


def test_8_reproducibility_random_seed() -> None:
    genders = ["Male"] * 51 + ["Female"] * 49 + [None] * 15
    df = pd.DataFrame({"Gender": genders})
    cleaned_a, _ = _impute(df.copy())
    cleaned_b, _ = _impute(df.copy())
    assert cleaned_a["Gender"].tolist() == cleaned_b["Gender"].tolist()


def test_original_dataset_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "original.csv"
    df = pd.DataFrame(
        {
            "City": ["Lahore", "Karachi", "Lahore", "Karachi"],
            "Gender": ["Male", "Female", "Male", None],
        }
    )
    df.to_csv(source, index=False)
    original_bytes = source.read_bytes()
    config = CleaningConfig(
        remove_duplicate_rows=False,
        handle_missing_values=True,
        handle_invalid_values=False,
        handle_empty_strings=False,
        handle_outliers=False,
        min_group_size=2,
        min_valid_observations=3,
    )
    cleaned, _, _ = run_cleaning_pipeline(pd.read_csv(source), config)
    assert source.read_bytes() == original_bytes
    assert cleaned["Gender"].isna().sum() == 0
