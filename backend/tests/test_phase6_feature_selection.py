"""Phase 6 feature selection unit + integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.session import Base
from app.models.dataset import Dataset
from app.schemas.cleaning import CleaningConfig, OutlierStrategy
from app.schemas.feature_selection import FeatureSelectionApplyRequest
from app.services.data_cleaner import apply_cleaning
from app.services.feature_decision_engine import (
    columns_to_drop_for_recommended_selection,
)
from app.services.feature_engineering_pipeline import apply_phase5_feature_engineering
from app.services.feature_quality_analyzer import analyze_feature_quality
from app.services.feature_redundancy_analyzer import analyze_correlation_pairs
from app.services.feature_selection_pipeline import (
    FeatureSelectionError,
    analyze_feature_selection,
    apply_feature_selection,
    run_feature_selection_analysis,
)
from app.services.feature_target_analyzer import (
    compute_mutual_information,
    detect_target_column,
    infer_target_task,
)
from app.services.feature_vif_analyzer import analyze_vif
from app.services.pipeline_state import (
    build_pipeline_status,
    feature_engineered_dataset_path,
    feature_selected_dataset_path,
    require_feature_engineered_dataframe,
)


def _phase6_dataframe(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    age = rng.integers(18, 70, n).astype(float)
    salary = rng.lognormal(10.5, 0.4, n)
    experience = rng.integers(0, 25, n).astype(float)
    experience_months = experience * 12 + rng.normal(0, 0.5, n)
    experience_days = experience * 365 + rng.normal(0, 2, n)
    churn = ((salary < np.median(salary)) & (experience < 8)).astype(int)

    return pd.DataFrame(
        {
            "Age": age,
            "Salary": salary,
            "Experience": experience,
            "Experience_Months": experience_months,
            "Experience_Days": experience_days,
            "Gender": rng.choice(["Male", "Female"], n),
            "City": rng.choice(
                [f"City_{i}" for i in range(n // 2)], n
            ),  # high cardinality
            "Is_Active": rng.choice([0, 1], n),
            "Customer_ID": [f"C{i:05d}" for i in range(n)],
            "Salary_Copy": salary.copy(),
            "Salary_Log": np.log1p(salary),
            "Review": [
                "This is a sufficiently long product review text for testing."
            ]
            * n,
            "Join_Date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
            "Constant_Column": ["SAME"] * n,
            "Near_Constant_Column": (
                ["A"] * int(np.floor(0.99 * n))
                + ["B"] * (n - int(np.floor(0.99 * n)))
            ),
            "Churn": churn,
        }
    )


def _make_db(tmp_path: Path, df: pd.DataFrame, filename: str = "phase6.csv"):
    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    raw_path = tmp_path / "upload.csv"
    df.to_csv(raw_path, index=False)
    dataset = Dataset(
        filename=filename,
        rows=len(df),
        columns=len(df.columns),
        file_size=raw_path.stat().st_size,
        dataset_path=str(raw_path),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return db, dataset, raw_path


def test_constant_near_constant_identifier_duplicate() -> None:
    df = _phase6_dataframe(200)
    quality = analyze_feature_quality(df, target_column="Churn")
    by = {q.feature: q for q in quality}
    assert by["Constant_Column"].is_constant
    assert by["Near_Constant_Column"].is_near_constant
    assert by["Customer_ID"].is_identifier
    assert by["Salary_Copy"].is_exact_duplicate
    assert by["Salary_Copy"].duplicate_of == "Salary"
    assert by["City"].semantic_type in {
        "high_cardinality_categorical",
        "categorical",
        "text",
    }


def test_correlation_and_vif_detect_redundancy() -> None:
    df = _phase6_dataframe(120)
    quality = analyze_feature_quality(df, target_column="Churn")
    pairs = analyze_correlation_pairs(df, quality_rows=quality, target_column="Churn")
    pair_names = {(p.feature_a, p.feature_b) for p in pairs} | {
        (p.feature_b, p.feature_a) for p in pairs
    }
    assert ("Salary", "Salary_Log") in pair_names or ("Salary_Log", "Salary") in pair_names

    vif_rows, available, msg = analyze_vif(
        df, quality_rows=quality, target_column="Churn", correlation_pairs=pairs
    )
    assert available is True
    assert msg is None
    high = [r for r in vif_rows if r.status == "HIGH"]
    assert high  # Experience family / salary family should elevate VIF


def test_no_target_pipeline_works() -> None:
    df = _phase6_dataframe(60).drop(columns=["Churn"])
    report = run_feature_selection_analysis(
        df, dataset_id=1, filename="x.csv", target_column=None
    )
    assert report.summary.target_column is None
    assert report.summary.target_aware_applied is False
    assert report.target_message
    assert report.decisions


def test_classification_and_regression_target() -> None:
    df = _phase6_dataframe(100)
    assert infer_target_task(df["Churn"]) == "classification"
    scores, err, mi_map = compute_mutual_information(df, "Churn")
    assert err is None or scores
    assert mi_map
    assert scores[0].target_type == "classification"

    df2 = df.copy()
    df2["Salary_Target"] = df2["Salary"]
    assert infer_target_task(df2["Salary_Target"]) == "regression"
    scores_r, err_r, _ = compute_mutual_information(
        df2.drop(columns=["Churn"]), "Salary_Target"
    )
    assert err_r is None or scores_r
    if scores_r:
        assert scores_r[0].target_type == "regression"


def test_target_protection_and_review_kept() -> None:
    df = _phase6_dataframe(200)
    report = run_feature_selection_analysis(
        df, dataset_id=1, filename="x.csv", target_column="Churn"
    )
    by = {d.feature: d for d in report.decisions}
    assert by["Churn"].decision == "KEEP"
    assert by["Churn"].is_target is True
    assert by["Constant_Column"].decision == "REMOVE"
    assert by["Customer_ID"].decision == "REMOVE"
    assert by["Salary_Copy"].decision == "REMOVE"
    assert by["Near_Constant_Column"].decision == "REVIEW"
    assert by["Is_Active"].decision in {"KEEP", "REVIEW"}

    drop = columns_to_drop_for_recommended_selection(
        report.decisions, target_column="Churn"
    )
    assert "Churn" not in drop
    assert "Constant_Column" in drop
    assert "Near_Constant_Column" not in drop  # REVIEW kept by default


def test_generated_feature_metadata_preserved(tmp_path: Path, monkeypatch) -> None:
    from app.core import config as cfg

    monkeypatch.setattr(cfg.settings, "BACKEND_DIR", tmp_path)
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)

    df = _phase6_dataframe(50)
    db, dataset, _ = _make_db(tmp_path, df, filename="meta_test.csv")

    # Write fake Phase 5 featured + metadata
    featured = feature_engineered_dataset_path(dataset.filename)
    featured.parent.mkdir(parents=True, exist_ok=True)
    # Add an engineered-looking column
    featured_df = df.copy()
    featured_df["Salary_Per_Experience"] = featured_df["Salary"] / (
        featured_df["Experience"] + 1
    )
    featured_df.to_csv(featured, index=False)

    from app.services.pipeline_state import save_feature_engineering_metadata

    save_feature_engineering_metadata(
        dataset.filename,
        {
            "generated_features": [
                {
                    "feature": "Salary_Per_Experience",
                    "source": "Salary + Experience",
                    "transformation": "Salary / (Experience + 1)",
                    "category": "numerical",
                    "phase": 5,
                }
            ]
        },
    )

    report = analyze_feature_selection(db, dataset.id)
    by = {d.feature: d for d in report.decisions}
    assert by["Salary_Per_Experience"].is_generated is True
    assert by["Salary_Per_Experience"].source_feature == "Salary + Experience"
    db.close()


def test_phase6_requires_feature_engineered(tmp_path: Path, monkeypatch) -> None:
    from app.core import config as cfg

    monkeypatch.setattr(cfg.settings, "BACKEND_DIR", tmp_path)
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)

    df = _phase6_dataframe(30)
    db, dataset, _ = _make_db(tmp_path, df)
    with pytest.raises(FeatureSelectionError) as exc:
        analyze_feature_selection(db, dataset.id)
    assert "Phase 5" in str(exc.value) or "feature-engineered" in str(exc.value).lower()
    db.close()


def test_apply_selection_preserves_rows_and_checkpoints(
    tmp_path: Path, monkeypatch
) -> None:
    from app.core import config as cfg

    monkeypatch.setattr(cfg.settings, "BACKEND_DIR", tmp_path)
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)

    df = _phase6_dataframe(60)
    db, dataset, raw_path = _make_db(tmp_path, df, filename="apply_test.csv")
    raw_bytes = raw_path.read_bytes()

    featured = feature_engineered_dataset_path(dataset.filename)
    featured.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(featured, index=False)
    featured_bytes = featured.read_bytes()

    result = apply_feature_selection(
        db,
        dataset.id,
        FeatureSelectionApplyRequest(target_column="Churn", apply_recommended=True),
    )
    assert result.applied is True
    assert result.before is not None and result.after is not None
    assert result.after.rows == result.before.rows
    assert result.after.columns < result.before.columns
    assert "Churn" in pd.read_csv(result.selected_path).columns
    assert raw_path.read_bytes() == raw_bytes
    assert featured.read_bytes() == featured_bytes
    assert feature_selected_dataset_path(dataset.filename).is_file()

    status = build_pipeline_status(db, dataset.id)
    assert status["current_stage"] == "feature_selected"
    assert status["feature_selected_available"] is True
    assert status["phase6_ready"] is True
    phase6 = next(p for p in status["phases"] if p["phase"] == 6)
    assert phase6["status"] == "complete"
    db.close()


def test_cumulative_phase4_5_6(tmp_path: Path, monkeypatch) -> None:
    from app.core import config as cfg

    monkeypatch.setattr(cfg.settings, "BACKEND_DIR", tmp_path)
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)

    df = _phase6_dataframe(80)
    db, dataset, raw_path = _make_db(tmp_path, df, filename="cumul.csv")
    raw_bytes = raw_path.read_bytes()

    config = CleaningConfig(
        remove_duplicate_rows=True,
        remove_duplicate_columns=False,  # keep Salary_Copy for Phase 6 tests
        remove_constant_columns=False,
        handle_missing_values=True,
        handle_invalid_values=True,
        handle_empty_strings=True,
        handle_outliers=True,
        outlier_strategy=OutlierStrategy.FLAG,
        convert_safe_dtypes=True,
        min_group_size=2,
    )
    apply_cleaning(db, dataset.id, config)
    apply_phase5_feature_engineering(db, dataset.id)

    featured_path = feature_engineered_dataset_path(dataset.filename)
    featured_bytes = featured_path.read_bytes()

    report = apply_feature_selection(
        db,
        dataset.id,
        FeatureSelectionApplyRequest(target_column="Churn"),
    )
    assert report.applied
    assert report.source == "feature_engineered"
    assert raw_path.read_bytes() == raw_bytes
    assert featured_path.read_bytes() == featured_bytes

    _, feat_df, _ = require_feature_engineered_dataframe(db, dataset.id)
    selected = pd.read_csv(report.selected_path)
    assert len(selected) == len(feat_df)
    assert set(selected.columns).issubset(set(feat_df.columns))
    db.close()
