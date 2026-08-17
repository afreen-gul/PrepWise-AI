"""Pipeline integration: Phase 4 cleaned checkpoint → Phase 5 (no raw fallback)."""

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
from app.services.data_cleaner import apply_cleaning, run_cleaning_pipeline
from app.services.feature_engineering_pipeline import (
    FeatureEngineeringPipelineError,
    apply_phase5_feature_engineering,
    run_feature_engineering_pipeline,
)
from app.services.pipeline_state import (
    PipelineStateError,
    build_pipeline_status,
    feature_engineered_dataset_path,
    load_feature_engineering_metadata,
    require_cleaned_dataframe,
    require_feature_engineered_dataframe,
)


def _messy_dataframe(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Customer_ID": [f"C{i:04d}" for i in range(n)],
            "Age": [None if i % 7 == 0 else int(rng.integers(18, 70)) for i in range(n)],
            "Salary": rng.lognormal(10.5, 1.1, n),
            "Gender": [None if i % 11 == 0 else ("Male" if i % 2 == 0 else "Female") for i in range(n)],
            "City": ["Lahore", "Karachi"] * (n // 2),
            "City_Copy": ["Lahore", "Karachi"] * (n // 2),
            "Join_Date": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
            "Review": [
                "This is a sufficiently long product review for testing."
            ]
            * n,
            "Constant_Column": ["SAME"] * n,
            "Years_Experience": rng.integers(1, 20, n).astype(float),
        }
    )


def test_phase5_requires_cleaned_checkpoint(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 't.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    raw_path = tmp_path / "upload.csv"
    df = _messy_dataframe(20)
    df.to_csv(raw_path, index=False)
    dataset = Dataset(
        filename="upload.csv",
        rows=len(df),
        columns=len(df.columns),
        file_size=raw_path.stat().st_size,
        dataset_path=str(raw_path),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    with pytest.raises(FeatureEngineeringPipelineError) as exc:
        apply_phase5_feature_engineering(db, dataset.id)
    assert "Phase 4" in str(exc.value) or "cleaned" in str(exc.value).lower()

    status = build_pipeline_status(db, dataset.id)
    assert status["current_stage"] == "raw"
    assert status["phase5_ready"] is False
    assert status["phase6_ready"] is False
    db.close()


def test_phase4_to_phase5_cumulative_persistence() -> None:
    raw = _messy_dataframe(60)
    original_cols = list(raw.columns)
    assert "Constant_Column" in original_cols
    assert "City_Copy" in original_cols

    config = CleaningConfig(
        remove_duplicate_rows=True,
        remove_duplicate_columns=True,
        remove_constant_columns=True,
        handle_missing_values=True,
        handle_invalid_values=True,
        handle_empty_strings=True,
        handle_outliers=True,
        outlier_strategy=OutlierStrategy.FLAG,
        convert_safe_dtypes=True,
        min_group_size=2,
        min_valid_observations=5,
    )
    cleaned, log, _ = run_cleaning_pipeline(raw.copy(), config)

    # Phase 4 effects
    assert "Constant_Column" not in cleaned.columns
    assert "City_Copy" not in cleaned.columns
    assert cleaned["Age"].isna().sum() == 0 or cleaned["Age"].isna().sum() < raw["Age"].isna().sum()
    cleaned_cols = list(cleaned.columns)
    cleaned_rows = len(cleaned)

    featured, details = run_feature_engineering_pipeline(cleaned)

    # Phase 4 removals persist
    assert "Constant_Column" not in featured.columns
    assert "City_Copy" not in featured.columns
    for col in cleaned_cols:
        assert col in featured.columns

    # Phase 5 additions
    assert any(c.startswith("Join_Date_") for c in featured.columns)
    assert len(featured) == cleaned_rows
    assert details["generated"]
    assert all(g.phase == 5 for g in details["generated"])

    # Raw unchanged
    assert list(raw.columns) == original_cols


def test_apply_phase5_uses_cleaned_not_raw(tmp_path: Path, monkeypatch) -> None:
    from app.core import config as cfg

    monkeypatch.setattr(cfg.settings, "BACKEND_DIR", tmp_path)
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{tmp_path / 't2.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    raw_path = tmp_path / "messy.csv"
    df = _messy_dataframe(50)
    df.to_csv(raw_path, index=False)
    raw_bytes = raw_path.read_bytes()

    dataset = Dataset(
        filename="messy.csv",
        rows=len(df),
        columns=len(df.columns),
        file_size=raw_path.stat().st_size,
        dataset_path=str(raw_path),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    result_clean = apply_cleaning(
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
    assert result_clean.cleaned_path
    cleaned_path = Path(result_clean.cleaned_path)
    cleaned_bytes = cleaned_path.read_bytes()
    cleaned_df = pd.read_csv(cleaned_path)
    assert "Constant_Column" not in cleaned_df.columns

    result_fe = apply_phase5_feature_engineering(db, dataset.id)
    assert result_fe.source == "cleaned"
    assert result_fe.applied_to_cleaned_dataset is True
    assert result_fe.pipeline_stage_before == "cleaned"
    assert result_fe.pipeline_stage_after == "feature_engineered"

    featured = pd.read_csv(result_fe.featured_path)
    assert "Constant_Column" not in featured.columns
    assert len(featured) == result_fe.before.rows == result_fe.after.rows
    assert raw_path.read_bytes() == raw_bytes
    assert cleaned_path.read_bytes() == cleaned_bytes

    meta = load_feature_engineering_metadata(dataset.filename)
    assert meta is not None
    assert meta["source_checkpoint"] == "cleaned"
    assert meta["phase"] == 5

    status = build_pipeline_status(db, dataset.id)
    assert status["current_stage"] == "feature_engineered"
    assert status["phase5_ready"] is True
    assert status["phase6_ready"] is True

    # Phase 6 hand-off helper works
    _ds, fe_df, _path = require_feature_engineered_dataframe(db, dataset.id)
    assert list(fe_df.columns) == list(featured.columns)

    from app.schemas.feature_selection import FeatureSelectionApplyRequest
    from app.services.feature_selection_pipeline import apply_feature_selection

    featured_bytes = Path(result_fe.featured_path).read_bytes()
    result_fs = apply_feature_selection(
        db,
        dataset.id,
        FeatureSelectionApplyRequest(apply_recommended=True),
    )
    assert result_fs.applied is True
    assert result_fs.after is not None
    assert result_fs.after.rows == result_fs.before.rows
    assert Path(result_fe.featured_path).read_bytes() == featured_bytes
    assert raw_path.read_bytes() == raw_bytes
    assert cleaned_path.read_bytes() == cleaned_bytes

    status_after = build_pipeline_status(db, dataset.id)
    assert status_after["current_stage"] == "feature_selected"
    assert status_after["feature_selected_available"] is True

    db.close()


def test_stress_phase4_then_phase5_then_phase6_cumulative() -> None:
    path = Path(__file__).resolve().parents[1] / "uploads" / "0f28ff04_prepwise_stress_test.csv"
    if not path.exists():
        paths = list(Path(__file__).resolve().parents[1].glob("uploads/*stress*.csv"))
        assert paths
        path = paths[0]

    raw_bytes = path.read_bytes()
    raw = pd.read_csv(path)
    cleaned, _, _ = run_cleaning_pipeline(
        raw.copy(),
        CleaningConfig(
            remove_duplicate_columns=True,
            remove_constant_columns=True,
            handle_missing_values=True,
            handle_outliers=True,
            outlier_strategy=OutlierStrategy.FLAG,
        ),
    )
    cleaned_cols = set(cleaned.columns)
    featured, details = run_feature_engineering_pipeline(cleaned)

    assert path.read_bytes() == raw_bytes
    assert len(featured) == len(cleaned)
    assert cleaned_cols.issubset(set(featured.columns))
    for gone in ("Constant_Column", "City_Copy"):
        if gone in raw.columns and gone not in cleaned.columns:
            assert gone not in featured.columns
    assert details["generated"] or details["skipped"]

    from app.services.feature_selection_pipeline import run_feature_selection_analysis

    report = run_feature_selection_analysis(
        featured,
        dataset_id=0,
        filename=path.name,
        target_column=None,
    )
    assert report.summary.total_features == len(featured.columns)
    assert report.decisions
    drop = [
        d.feature for d in report.decisions if d.decision == "REMOVE"
    ]
    selected = featured.drop(columns=drop, errors="ignore")
    assert len(selected) == len(featured)
    assert len(selected.columns) <= len(featured.columns)
