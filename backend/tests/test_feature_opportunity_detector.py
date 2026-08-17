"""Tests for Phase 5.1 feature type & opportunity detection (analysis only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.services.feature_opportunity_detector import (
    detect_feature_relationships,
    detect_feature_type,
    generate_feature_engineering_opportunities,
    generate_feature_engineering_report,
)


def _by_column(analyses: list) -> dict:
    return {a.column: a for a in analyses}


def test_1_numerical_age_salary() -> None:
    df = pd.DataFrame(
        {
            "Age": [22, 35, 41, 28, 55],
            "Salary": [40_000, 55_000, 62_000, 48_000, 90_000],
        }
    )
    analyses, _, _ = generate_feature_engineering_opportunities(df)
    by_col = _by_column(analyses)
    assert by_col["Age"].detected_type == "numerical"
    assert by_col["Salary"].detected_type == "numerical"


def test_2_categorical_gender_city() -> None:
    df = pd.DataFrame(
        {
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
            "City": ["Lahore", "Karachi", "Lahore", "Islamabad", "Karachi"],
        }
    )
    analyses, _, _ = generate_feature_engineering_opportunities(df)
    by_col = _by_column(analyses)
    assert by_col["Gender"].detected_type == "categorical"
    assert by_col["City"].detected_type == "categorical"


def test_3_datetime_join_date_opportunity() -> None:
    df = pd.DataFrame(
        {
            "Join_Date": [
                "2024-01-15",
                "2023-05-21",
                "2025-02-01",
                "2022-11-30",
                "2024-07-04",
            ]
        }
    )
    analyses, _, _ = generate_feature_engineering_opportunities(df)
    join = _by_column(analyses)["Join_Date"]
    assert join.detected_type == "datetime"
    assert "datetime" in join.opportunity.lower() or "decomposition" in join.opportunity.lower()
    assert join.priority == "HIGH"


def test_4_identifier_customer_id_no_numeric_fe() -> None:
    df = pd.DataFrame(
        {
            "Customer_ID": [f"C{i:04d}" for i in range(50)],
            "Value": list(range(50)),
        }
    )
    analyses, _, _ = generate_feature_engineering_opportunities(df)
    cust = _by_column(analyses)["Customer_ID"]
    assert cust.detected_type == "identifier"
    assert cust.opportunity.lower().startswith("none")
    assert "identifier" in cust.reason.lower()


def test_5_boolean_is_active() -> None:
    df = pd.DataFrame({"Is_Active": ["Yes", "No", "Yes", "Yes", "No"]})
    ftype, _ = detect_feature_type(df["Is_Active"], "Is_Active")
    assert ftype == "boolean"


def test_6_text_review_opportunity() -> None:
    reviews = [
        "The product arrived late but customer support was helpful and resolved my issue quickly.",
        "Excellent quality and packaging. Would definitely recommend to friends and family.",
        "Average experience overall. The description did not match the item I received.",
        "Terrible service. I waited two weeks and still have not received a refund response.",
        "Great value for money. The materials feel premium and shipping was surprisingly fast.",
    ]
    df = pd.DataFrame({"Review": reviews})
    analyses, _, _ = generate_feature_engineering_opportunities(df)
    review = _by_column(analyses)["Review"]
    assert review.detected_type == "text"
    assert "text" in review.opportunity.lower()


def test_7_skewed_salary_log_recommendation_no_transform() -> None:
    rng = np.random.default_rng(42)
    # Strong right skew via lognormal
    salary = rng.lognormal(mean=10.5, sigma=1.2, size=200)
    df = pd.DataFrame({"Salary": salary})
    before = df.copy()
    analyses, _, _ = generate_feature_engineering_opportunities(df)
    sal = _by_column(analyses)["Salary"]
    assert sal.detected_type == "numerical"
    assert "log" in sal.opportunity.lower() or "skew" in sal.opportunity.lower()
    # No transform
    assert df["Salary"].tolist() == before["Salary"].tolist()
    assert list(df.columns) == list(before.columns)


def test_8_constant_near_constant() -> None:
    df = pd.DataFrame(
        {
            "Constant_Column": ["Pakistan"] * 100,
            "Near_Constant": ["A"] * 99 + ["B"],
        }
    )
    analyses, _, _ = generate_feature_engineering_opportunities(df)
    by_col = _by_column(analyses)
    assert by_col["Constant_Column"].detected_type == "constant / near-constant"
    assert by_col["Near_Constant"].detected_type == "constant / near-constant"
    assert "low-information" in by_col["Constant_Column"].opportunity.lower()


def test_9_height_weight_bmi_relationship_not_created() -> None:
    df = pd.DataFrame(
        {
            "Height": [1.70, 1.65, 1.80, 1.75, 1.60],
            "Weight": [70, 62, 85, 78, 55],
        }
    )
    before_cols = list(df.columns)
    analyses, relationships, _ = generate_feature_engineering_opportunities(df)
    type_map = {a.column: a.detected_type for a in analyses}
    assert type_map["Height"] == "numerical"
    assert type_map["Weight"] == "numerical"
    assert any("BMI" in r.opportunity or "bmi" in r.opportunity.lower() for r in relationships)
    assert list(df.columns) == before_cols
    assert "BMI" not in df.columns


def test_10_customer_id_no_artificial_opportunity() -> None:
    df = pd.DataFrame({"Customer_ID": list(range(1001, 1101))})
    analyses, _, _ = generate_feature_engineering_opportunities(df)
    cust = _by_column(analyses)["Customer_ID"]
    assert cust.detected_type == "identifier"
    assert cust.opportunity.lower().startswith("none")
    assert not cust.opportunities


def test_data_integrity_report_unchanged() -> None:
    df = pd.DataFrame(
        {
            "Age": [20, 30, 40],
            "Gender": ["M", "F", "M"],
            "Customer_ID": ["a", "b", "c"],
        }
    )
    original = df.copy(deep=True)
    report = generate_feature_engineering_report(df, dataset_id=1, source="original")
    assert report.transformations_applied is False
    assert report.column_count_unchanged is True
    assert list(df.columns) == list(original.columns)
    assert df.equals(original)


def test_stress_csv_feature_types_no_transform() -> None:
    path = Path(__file__).resolve().parents[1] / "uploads" / "0f28ff04_prepwise_stress_test.csv"
    if not path.exists():
        paths = list(Path(__file__).resolve().parents[1].glob("uploads/*stress*.csv"))
        assert paths, "stress test CSV not found"
        path = paths[0]

    original_bytes = path.read_bytes()
    df = pd.read_csv(path)
    before_cols = list(df.columns)
    before = df.copy(deep=True)

    report = generate_feature_engineering_report(df, dataset_id=0, source="original")
    by_col = _by_column(report.column_analyses)

    assert by_col["Customer_ID"].detected_type == "identifier"
    assert by_col["Gender"].detected_type == "categorical"
    assert by_col["City"].detected_type == "categorical"
    assert by_col["Age"].detected_type == "numerical"
    assert by_col["Salary"].detected_type == "numerical"
    assert by_col["Join_Date"].detected_type == "datetime"
    assert "datetime" in by_col["Join_Date"].opportunity.lower() or "decomposition" in by_col[
        "Join_Date"
    ].opportunity.lower()
    assert by_col["Constant_Column"].detected_type == "constant / near-constant"
    assert "Churn" in report.potential_targets
    assert report.transformations_applied is False
    assert list(df.columns) == before_cols
    assert df.equals(before)
    assert path.read_bytes() == original_bytes


def test_detect_feature_relationships_helper() -> None:
    df = pd.DataFrame({"Height_cm": [170, 180], "Weight_kg": [70, 80]})
    type_map = {"Height_cm": "numerical", "Weight_kg": "numerical"}
    rels = detect_feature_relationships(df, type_map)
    assert rels
    assert any("BMI" in r.opportunity for r in rels)
