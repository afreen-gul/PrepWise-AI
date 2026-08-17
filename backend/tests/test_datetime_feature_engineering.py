"""Tests for Phase 5.2 datetime feature engineering."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from app.services.datetime_feature_engineer import engineer_datetime_features


def _generated_names(details: dict) -> set[str]:
    return {g.feature for g in details["generated"]}


def _skipped_names(details: dict) -> set[str]:
    return {s.feature for s in details["skipped"]}


def test_1_basic_date_extraction() -> None:
    df = pd.DataFrame(
        {
            "Join_Date": ["2024-01-15", "2024-02-20", "2025-03-10"],
        }
    )
    out, details = engineer_datetime_features(df)
    names = _generated_names(details)
    assert "Join_Date_Year" in names
    assert "Join_Date_Month" in names
    assert "Join_Date_Day" in names
    assert "Join_Date_DayOfWeek" in names
    assert "Join_Date" in out.columns
    assert list(out["Join_Date_Year"]) == [2024, 2024, 2025]


def test_2_weekend_detection() -> None:
    # 2024-01-15 = Monday, 2024-01-20 = Saturday
    df = pd.DataFrame({"Join_Date": ["2024-01-15", "2024-01-20", "2024-01-21"]})
    out, details = engineer_datetime_features(df)
    assert "Join_Date_IsWeekend" in _generated_names(details)
    assert set(out["Join_Date_IsWeekend"].dropna().astype(int)) == {0, 1}
    assert int(out.loc[0, "Join_Date_IsWeekend"]) == 0
    assert int(out.loc[1, "Join_Date_IsWeekend"]) == 1


def test_3_constant_year_skipped() -> None:
    df = pd.DataFrame(
        {"Join_Date": ["2025-01-15", "2025-06-20", "2025-12-01"]}
    )
    out, details = engineer_datetime_features(df)
    assert "Join_Date_Year" not in out.columns
    assert "Join_Date_Year" in _skipped_names(details)
    reason = next(s.reason for s in details["skipped"] if s.feature == "Join_Date_Year")
    assert "same year" in reason.lower()


def test_4_constant_weekend_skipped() -> None:
    # Mon / Tue / Wed only
    df = pd.DataFrame(
        {"Join_Date": ["2024-01-15", "2024-01-16", "2024-01-17"]}
    )
    out, details = engineer_datetime_features(df)
    assert "Join_Date_IsWeekend" not in out.columns
    assert "Join_Date_IsWeekend" in _skipped_names(details)
    reason = next(
        s.reason for s in details["skipped"] if s.feature == "Join_Date_IsWeekend"
    )
    assert "weekday" in reason.lower()


def test_5_timestamp_hour_minute() -> None:
    df = pd.DataFrame(
        {
            "Event_Time": [
                "2025-01-15 14:35:00",
                "2025-01-15 09:10:00",
                "2025-02-01 18:00:00",
            ]
        }
    )
    out, details = engineer_datetime_features(df)
    names = _generated_names(details)
    assert "Event_Time_Hour" in names
    assert "Event_Time_Minute" in names
    assert "Event_Time_Second" not in out.columns
    assert "Event_Time_Second" in _skipped_names(details)


def test_6_constant_midnight_no_time_features() -> None:
    df = pd.DataFrame(
        {
            "Join_Date": [
                "2024-01-15 00:00:00",
                "2024-02-20 00:00:00",
                "2025-03-10 00:00:00",
            ]
        }
    )
    out, details = engineer_datetime_features(df)
    assert "Join_Date_Hour" not in out.columns
    assert "Join_Date_Minute" not in out.columns
    assert any("Hour" in s.feature for s in details["skipped"])


def test_7_start_end_duration() -> None:
    df = pd.DataFrame(
        {
            "Start_Date": ["2024-01-01", "2024-01-01"],
            "End_Date": ["2024-01-15", "2024-01-11"],
        }
    )
    out, details = engineer_datetime_features(df)
    feat = "Start_Date_End_Date_Duration_Days"
    assert feat in _generated_names(details)
    assert float(out.loc[0, feat]) == 14.0
    assert float(out.loc[1, feat]) == 10.0


def test_8_invalid_date_no_crash() -> None:
    df = pd.DataFrame(
        {
            "Join_Date": [
                "2024-01-15",
                "not-a-date",
                "2025-03-10",
                "2023-06-01",
            ]
        }
    )
    out, details = engineer_datetime_features(df)
    assert len(out) == 4
    assert any(i.issue == "invalid_datetime" for i in details["issues"])
    assert "Join_Date_Year" in out.columns
    assert pd.isna(out.loc[1, "Join_Date_Year"])
    assert int(out.loc[0, "Join_Date_Year"]) == 2024
    assert int(out.loc[2, "Join_Date_Year"]) == 2025


def test_9_birth_after_reference_flagged() -> None:
    df = pd.DataFrame(
        {
            "Date_of_Birth": ["2000-01-01", "2030-01-01"],
            "Reference_Date": ["2020-01-01", "2020-01-01"],
        }
    )
    out, details = engineer_datetime_features(df)
    assert any(i.issue == "birth_after_reference" for i in details["issues"])
    # Dates not modified
    assert list(out["Date_of_Birth"]) == list(df["Date_of_Birth"])
    assert list(out["Reference_Date"]) == list(df["Reference_Date"])


def test_10_existing_feature_not_duplicated() -> None:
    df = pd.DataFrame(
        {
            "Join_Date": ["2024-01-15", "2025-02-20", "2024-03-10"],
            "Join_Date_Year": [9999, 9999, 9999],
        }
    )
    out, details = engineer_datetime_features(df)
    assert list(out["Join_Date_Year"]) == [9999, 9999, 9999]
    assert any(
        s.feature == "Join_Date_Year" and "already exists" in s.reason.lower()
        for s in details["skipped"]
    )
    assert "Join_Date_Year_1" not in out.columns


def test_11_no_row_changes() -> None:
    df = pd.DataFrame(
        {
            "Join_Date": pd.date_range("2020-01-01", periods=2000, freq="D").astype(str),
            "Value": range(2000),
        }
    )
    out, _ = engineer_datetime_features(df)
    assert len(out) == 2000
    assert len(df) == 2000


def test_12_large_dataset_performance() -> None:
    n = 5000
    df = pd.DataFrame(
        {
            "Join_Date": pd.date_range("2018-01-01", periods=n, freq="h").astype(str),
        }
    )
    start = time.perf_counter()
    out, details = engineer_datetime_features(df)
    elapsed = time.perf_counter() - start
    assert len(out) == n
    assert details["generated"]
    assert elapsed < 10.0


def test_stress_csv_datetime_engineering() -> None:
    path = Path(__file__).resolve().parents[1] / "uploads" / "0f28ff04_prepwise_stress_test.csv"
    if not path.exists():
        paths = list(Path(__file__).resolve().parents[1].glob("uploads/*stress*.csv"))
        assert paths, "stress CSV missing"
        path = paths[0]

    original_bytes = path.read_bytes()
    df = pd.read_csv(path)
    before_cols = list(df.columns)
    before_rows = len(df)

    out, details = engineer_datetime_features(df)

    assert path.read_bytes() == original_bytes
    assert list(df.columns) == before_cols
    assert len(df) == before_rows
    assert len(out) == before_rows
    assert "Join_Date" in out.columns
    assert "Join_Date" in details["datetime_columns"]
    # Original columns still present
    for col in before_cols:
        assert col in out.columns
    assert details["generated"] or details["skipped"]
