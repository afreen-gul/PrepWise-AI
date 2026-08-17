"""Phase 5 selective / optional feature engineering."""

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
from app.schemas.feature_engineering import FeatureEngineeringApplyRequest
from app.services.data_cleaner import apply_cleaning
from app.services.feature_engineering_pipeline import (
    apply_phase5_feature_engineering,
    discover_feature_candidates,
    run_feature_engineering_pipeline,
)
from app.services.pipeline_state import (
    build_pipeline_status,
    feature_engineered_dataset_path,
)


def _sample_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Age": rng.integers(18, 70, n).astype(float),
            "Salary": rng.lognormal(10.5, 0.5, n),
            "Years_Experience": rng.integers(1, 20, n).astype(float),
            "Join_Date": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
            "Review": ["Great product with solid quality overall."] * n,
            "City": ["Lahore", "Karachi"] * (n // 2),
        }
    )


def test_full_auto_still_works_when_selection_is_none() -> None:
    df = _sample_df(50)
    out, details = run_feature_engineering_pipeline(df, selected_feature_ids=None)
    assert len(out) == len(df)
    assert details["generated"]
    assert details["selection_mode"] == "all"
    for col in df.columns:
        assert col in out.columns


def test_pass_through_empty_selection() -> None:
    df = _sample_df(30)
    out, details = run_feature_engineering_pipeline(df, selected_feature_ids=[])
    assert list(out.columns) == list(df.columns)
    assert details["generated"] == []
    assert details["selection_mode"] == "pass_through"
    assert len(out) == len(df)


def test_selective_generation_only_creates_chosen_features() -> None:
    df = _sample_df(50)
    _, all_details = run_feature_engineering_pipeline(df, selected_feature_ids=None)
    all_names = [g.feature for g in all_details["generated"]]
    assert all_names
    chosen = all_names[:2]
    out, details = run_feature_engineering_pipeline(df, selected_feature_ids=chosen)
    created = {g.feature for g in details["generated"]}
    assert created.issubset(set(chosen))
    for name in chosen:
        if name in created:
            assert name in out.columns
    # Non-selected engineered names must not appear
    for name in all_names:
        if name not in chosen:
            assert name not in out.columns
    for col in df.columns:
        assert col in out.columns


def test_discover_and_apply_selective(tmp_path: Path, monkeypatch) -> None:
    from app.core import config as cfg

    monkeypatch.setattr(cfg.settings, "BACKEND_DIR", tmp_path)
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    df = _sample_df(40)
    raw = tmp_path / "upload.csv"
    df.to_csv(raw, index=False)
    dataset = Dataset(
        filename="sel_fe.csv",
        rows=len(df),
        columns=len(df.columns),
        file_size=raw.stat().st_size,
        dataset_path=str(raw),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    apply_cleaning(
        db,
        dataset.id,
        CleaningConfig(
            remove_duplicate_columns=True,
            remove_constant_columns=True,
            handle_missing_values=True,
            handle_outliers=True,
            outlier_strategy=OutlierStrategy.FLAG,
            min_group_size=2,
            min_valid_observations=5,
        ),
    )

    report = discover_feature_candidates(db, dataset.id)
    assert report.transformations_applied is False
    assert report.source == "cleaned"

    # Empty selection → pass-through featured checkpoint
    result = apply_phase5_feature_engineering(
        db,
        dataset.id,
        FeatureEngineeringApplyRequest(selected_feature_ids=[]),
    )
    assert result.selection_mode == "pass_through"
    assert result.features_generated == 0
    featured = pd.read_csv(result.featured_path)
    assert len(featured) == result.before.rows
    status = build_pipeline_status(db, dataset.id)
    assert status["phase6_ready"] is True
    assert status["feature_engineered_available"] is True

    # Selective apply
    if report.candidates:
        ids = [c.id for c in report.candidates[:1]]
        result2 = apply_phase5_feature_engineering(
            db,
            dataset.id,
            FeatureEngineeringApplyRequest(selected_feature_ids=ids),
        )
        assert result2.selection_mode == "selected"
        assert result2.features_generated <= 1
        featured2 = pd.read_csv(result2.featured_path)
        assert len(featured2) == result2.before.rows

    db.close()
