"""Unit tests for Phase 4 intelligent data cleaning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from app.schemas.cleaning import CleaningConfig, OutlierStrategy
from app.services.data_cleaner import (
    CategoricalImputationParams,
    convert_safe_dtypes,
    handle_invalid_numeric_values,
    handle_outliers,
    impute_missing_values,
    normalize_empty_strings,
    remove_duplicate_rows,
    run_cleaning_pipeline,
    _constant_columns,
)

TEST_CAT_PARAMS = CategoricalImputationParams(
    min_group_size=2,
    min_valid_observations=3,
)


@pytest.fixture
def cleaning_log() -> list:
    return []


def test_duplicate_removal(cleaning_log: list) -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    cleaned = remove_duplicate_rows(df, cleaning_log)
    assert len(cleaned) == 2
    assert cleaning_log[-1].details["rows_removed"] == 1
    # Original unchanged
    assert len(df) == 3


def test_numeric_missing_imputation_median_for_skew() -> None:
    # Highly skewed numeric series (≥8 values) → median preferred
    values = [1, 1, 1, 1, 2, 2, 3, 1000, None]
    df = pd.DataFrame({"score": values})
    log: list = []
    cleaned = impute_missing_values(
        df,
        log,
        high_missing_threshold=0.7,
        drop_high_missing=False,
        categorical_params=TEST_CAT_PARAMS,
    )
    assert cleaned["score"].isna().sum() == 0
    entry = next(e for e in log if e.operation == "missing_value_imputation")
    assert entry.details["method"] == "median"
    assert df["score"].isna().sum() == 1  # original untouched


def test_categorical_missing_imputation_mode() -> None:
    """Clear dominant category → Global Mode imputation."""
    df = pd.DataFrame({"Gender": ["Male", "Male", "Male", "Female", None]})
    log: list = []
    cleaned = impute_missing_values(
        df,
        log,
        high_missing_threshold=0.7,
        drop_high_missing=False,
        categorical_params=TEST_CAT_PARAMS,
    )
    assert cleaned["Gender"].isna().sum() == 0
    entry = next(e for e in log if e.operation == "missing_value_imputation")
    assert entry.details["method"] == "global_mode"
    assert entry.details["confidence"] == "high"
    assert cleaned["Gender"].iloc[4] == "Male"


def test_categorical_nearly_uniform_uses_distribution() -> None:
    df = pd.DataFrame(
        {
            "color": [
                "red",
                "blue",
                "green",
                "red",
                "blue",
                "green",
                "red",
                "blue",
                None,
                None,
            ]
        }
    )
    log: list = []
    cleaned = impute_missing_values(
        df,
        log,
        high_missing_threshold=0.7,
        drop_high_missing=False,
        categorical_params=TEST_CAT_PARAMS,
    )
    entry = next(e for e in log if e.operation == "missing_value_imputation")
    assert entry.details["method"] == "distribution_based"
    assert cleaned["color"].isna().sum() == 0


def test_categorical_elevated_missing_uses_global_mode_when_dominant() -> None:
    values = ["A"] * 7 + ["B"] * 2 + [None] * 5
    df = pd.DataFrame({"seg": values})
    log: list = []
    cleaned = impute_missing_values(
        df,
        log,
        high_missing_threshold=0.7,
        drop_high_missing=False,
        categorical_params=TEST_CAT_PARAMS,
    )
    entry = next(e for e in log if e.operation == "missing_value_imputation")
    assert entry.details["method"] == "global_mode"
    assert (cleaned["seg"] == "A").sum() >= 5


def test_categorical_too_few_non_null_flagged_for_review() -> None:
    df = pd.DataFrame({"seg": ["A", "B", None, None, None, None]})
    log: list = []
    cleaned = impute_missing_values(
        df,
        log,
        high_missing_threshold=0.7,
        drop_high_missing=False,
        categorical_params=TEST_CAT_PARAMS,
    )
    assert any(e.operation == "categorical_imputation_review" for e in log)
    assert cleaned["seg"].isna().sum() == 4


def test_empty_string_handling(cleaning_log: list) -> None:
    df = pd.DataFrame({"name": ["Alice", "", "  ", "Bob"]})
    cleaned = normalize_empty_strings(df, cleaning_log)
    assert cleaned["name"].isna().sum() == 2
    assert df["name"].isna().sum() == 0


def test_safe_dtype_conversion(cleaning_log: list) -> None:
    df = pd.DataFrame(
        {
            "age": ["25", "30", "40"],
            "note": ["a", "b", "c"],  # should NOT become numeric
        }
    )
    cleaned = convert_safe_dtypes(df, cleaning_log)
    assert pd.api.types.is_numeric_dtype(cleaned["age"])
    assert not pd.api.types.is_numeric_dtype(cleaned["note"])
    conversions = [e for e in cleaning_log if e.operation == "dtype_conversion"]
    assert any(e.column == "age" for e in conversions)
    assert not any(e.column == "note" for e in conversions)


def test_outlier_detection_flag() -> None:
    # Clear outliers with enough samples for IQR
    values = list(range(1, 21)) + [1000]
    df = pd.DataFrame({"x": values})
    log: list = []
    cleaned = handle_outliers(df, log, OutlierStrategy.FLAG)
    assert len(cleaned) == len(df)  # FLAG keeps rows
    assert any(e.operation == "statistical_outlier_flagged" for e in log)
    assert all(
        e.details.get("value_category") == "statistical_outlier"
        for e in log
        if e.operation == "statistical_outlier_flagged"
    )
    assert 1000 in cleaned["x"].values


def test_invalid_age_converted_then_imputed_not_as_outlier() -> None:
    """Impossible ages become missing + imputed; IQR outliers stay flagged."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 28, 32, 27, 29, 31, 26, -5, 999],
            "score": list(range(1, 11)),  # no domain rule;  may have mild spread
        }
    )
    config = CleaningConfig(
        remove_duplicate_rows=False,
        convert_safe_dtypes=True,
        handle_missing_values=True,
        handle_invalid_values=True,
        handle_empty_strings=True,
        handle_outliers=True,
        outlier_strategy=OutlierStrategy.FLAG,
        remove_constant_columns=False,
    )
    cleaned, log, _ = run_cleaning_pipeline(df, config)

    invalid_logs = [e for e in log if e.operation == "invalid_value_handling"]
    assert invalid_logs
    assert invalid_logs[0].details["value_category"] == "invalid_domain"
    assert invalid_logs[0].details["before_invalid"] >= 2
    assert "transformations" in invalid_logs[0].details

    impute_logs = [
        e
        for e in log
        if e.operation == "missing_value_imputation" and e.column == "age"
    ]
    assert impute_logs
    assert impute_logs[0].details.get("from_invalid_domain_count", 0) >= 2
    assert cleaned["age"].isna().sum() == 0
    # Impossible ages should not remain in the cleaned column
    assert (cleaned["age"] < 0).sum() == 0
    assert (cleaned["age"] > 120).sum() == 0

    # Statistical outlier flags (if any) must be labeled distinctly
    for entry in log:
        if entry.details.get("value_category") == "statistical_outlier":
            assert entry.operation.startswith("statistical_outlier")


def test_invalid_age_logs_before_after_values() -> None:
    df = pd.DataFrame({"age": [20, 25, 30, 35, 40, 45, 50, 55, -3]})
    log: list = []
    cleaned, indices = handle_invalid_numeric_values(df, log)
    assert cleaned["age"].isna().sum() == 1
    assert "age" in indices
    entry = log[0]
    assert entry.details["before_values"]
    assert -3 in entry.details["before_values"] or -3.0 in entry.details["before_values"]
    assert entry.details["after_value"] is None
    samples = entry.details["transformations"]
    assert samples[0]["before"] in (-3, -3.0)
    assert samples[0]["after"] is None


def test_constant_column_detection() -> None:
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    assert _constant_columns(df) == ["a"]


def test_original_dataset_remains_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "original.csv"
    df = pd.DataFrame(
        {
            "Customer_ID": [1, 2, 2],
            "age": ["25", "30", "30"],
            "city": ["NYC", "", "NYC"],
            "salary": [50000, -10, 50000],
        }
    )
    df.to_csv(source, index=False)
    original_bytes = source.read_bytes()

    loaded = pd.read_csv(source)
    config = CleaningConfig(
        remove_duplicate_rows=True,
        convert_safe_dtypes=True,
        handle_missing_values=True,
        handle_invalid_values=True,
        handle_empty_strings=True,
        handle_outliers=True,
        outlier_strategy=OutlierStrategy.FLAG,
        remove_constant_columns=False,
    )
    cleaned, log, _ = run_cleaning_pipeline(loaded, config)

    assert source.read_bytes() == original_bytes
    assert len(cleaned) <= len(loaded)
    assert any(e.operation == "duplicate_row_removal" for e in log)
    # ID column retained
    assert "Customer_ID" in cleaned.columns


def test_high_missing_column_review_not_dropped() -> None:
    df = pd.DataFrame(
        {
            "ok": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "sparse": [1] + [None] * 9,
        }
    )
    log: list = []
    cleaned = impute_missing_values(
        df,
        log,
        high_missing_threshold=0.7,
        drop_high_missing=False,
        categorical_params=TEST_CAT_PARAMS,
    )
    assert "sparse" in cleaned.columns
    assert any(e.operation == "high_missing_column_review" for e in log)


def test_integer_like_imputation_restores_int_dtype() -> None:
    """Age-like columns must not keep float fills such as 33.009."""
    df = pd.DataFrame(
        {
            "Age": [25, 31, None, 28, 30, 29, 27, 26, 32, 24],
            "Salary": [50000.5, 61000.25, None, 55000.0, 52000.1, 53000.2, 54000.3, 56000.4, 57000.6, 58000.7],
        }
    )
    log: list = []
    cleaned = impute_missing_values(
        df,
        log,
        high_missing_threshold=0.7,
        drop_high_missing=False,
        categorical_params=TEST_CAT_PARAMS,
    )

    assert pd.api.types.is_integer_dtype(cleaned["Age"])
    assert cleaned["Age"].isna().sum() == 0
    age_fill = cleaned.loc[df["Age"].isna(), "Age"].iloc[0]
    assert isinstance(age_fill, (int, np.integer)) or (
        hasattr(age_fill, "item") and float(age_fill).is_integer()
    )
    assert float(age_fill) == int(age_fill)

    age_log = next(e for e in log if e.column == "Age")
    assert age_log.details["method"] in {"mean", "median"}
    assert "Int64" in str(age_log.details["final_dtype"])
    assert isinstance(age_log.details["fill_value"], int)

    # Continuous salary with fractional values stays float
    assert pd.api.types.is_float_dtype(cleaned["Salary"])
    salary_log = next(e for e in log if e.column == "Salary")
    assert "float" in str(salary_log.details["final_dtype"]).lower()


def test_integer_like_after_invalid_age_pipeline() -> None:
    df = pd.DataFrame(
        {
            "Age": [25, 31, 28, 30, 29, 27, 26, 32, 24, -5],
        }
    )
    config = CleaningConfig(
        remove_duplicate_rows=False,
        convert_safe_dtypes=True,
        handle_missing_values=True,
        handle_invalid_values=True,
        handle_empty_strings=False,
        handle_outliers=False,
        remove_constant_columns=False,
    )
    cleaned, log, _ = run_cleaning_pipeline(df, config)
    assert pd.api.types.is_integer_dtype(cleaned["Age"])
    assert cleaned["Age"].isna().sum() == 0
    assert (cleaned["Age"] < 0).sum() == 0
    # No trailing float representation in values
    assert all(float(v).is_integer() for v in cleaned["Age"].tolist())
    impute = next(e for e in log if e.operation == "missing_value_imputation")
    assert "Int64" in str(impute.details["final_dtype"])
    assert impute.details["method"] in {"mean", "median"}

