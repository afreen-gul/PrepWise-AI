"""Phase 5.4 — Lightweight text feature engineering (length signals only).



No TF-IDF, embeddings, sentiment, or NLP models.

"""



from __future__ import annotations



from typing import Any



import pandas as pd



from app.schemas.feature_engineering import GeneratedFeatureMeta, SkippedFeatureMeta

from app.services.feature_engineering_config import MIN_NON_NULL_FOR_TRANSFORM

from app.services.feature_opportunity_detector import (

    detect_feature_type,

    detect_identifier,

    detect_text_feature,

)





class TextFeatureEngineeringError(Exception):

    """Raised for expected text FE failures."""





def _normalize_text(series: pd.Series) -> pd.Series:

    """NaN / empty / whitespace-only → empty string for counting."""

    text = series.astype("string")

    text = text.fillna("")

    text = text.str.strip()

    return text





def _has_variation(values: pd.Series) -> bool:

    return int(values.dropna().nunique()) > 1





def _try_add(

    working: pd.DataFrame,

    *,

    feature_name: str,

    values: pd.Series,

    source: str,

    transformation: str,

    reason: str,

    generated: list[GeneratedFeatureMeta],

    skipped: list[SkippedFeatureMeta],

    selected: set[str] | None = None,

) -> None:

    if selected is not None and feature_name not in selected:

        skipped.append(

            SkippedFeatureMeta(

                feature=feature_name,

                source=source,

                reason="Not selected by user.",

                category="text",

            )

        )

        return

    if feature_name in working.columns:

        skipped.append(

            SkippedFeatureMeta(

                feature=feature_name,

                source=source,

                reason="Feature already exists — skipped.",

                category="text",

            )

        )

        return

    if not _has_variation(values):

        skipped.append(

            SkippedFeatureMeta(

                feature=feature_name,

                source=source,

                reason="Skipped because the candidate text feature would be constant.",

                category="text",

            )

        )

        return

    working[feature_name] = values

    generated.append(

        GeneratedFeatureMeta(

            feature=feature_name,

            source=source,

            feature_type="Integer",

            category="text",

            transformation=transformation,

            reason=reason,

            rows_affected=int(values.notna().sum()),

            status="Created",

        )

    )





def engineer_text_features(

    df: pd.DataFrame,

    *,

    selected: set[str] | None = None,

) -> tuple[pd.DataFrame, dict[str, Any]]:

    """Create CharCount / WordCount for meaningful free-text columns."""

    working = df.copy()

    before_rows = len(working)

    generated: list[GeneratedFeatureMeta] = []

    skipped: list[SkippedFeatureMeta] = []



    for column in list(working.columns):

        col = str(column)

        ftype, details = detect_feature_type(working[col], col)



        if detect_identifier(working[col], col) or ftype == "identifier":

            skipped.append(

                SkippedFeatureMeta(

                    feature=f"{col}_WordCount",

                    source=col,

                    reason="Skipped: column appears to be an identifier.",

                    category="text",

                )

            )

            continue



        text_info = detect_text_feature(working[col], col)

        if ftype != "text" and not text_info:

            continue



        # Skip person-name columns for length FE (low modeling utility here)

        kind = (text_info or details).get("text_kind", "")

        if kind == "names":

            skipped.append(

                SkippedFeatureMeta(

                    feature=f"{col}_WordCount",

                    source=col,

                    reason="Skipped: person-name column — length features not applied.",

                    category="text",

                )

            )

            continue



        text = _normalize_text(working[col])

        valid_text = text.ne("")

        if int(valid_text.sum()) < MIN_NON_NULL_FOR_TRANSFORM:

            skipped.append(

                SkippedFeatureMeta(

                    feature=f"{col}_CharCount",

                    source=col,

                    reason="Skipped: insufficient valid text values.",

                    category="text",

                )

            )

            continue



        char_count = text.str.len().fillna(0).astype("Int64")

        word_lists = text.str.split()

        word_count = word_lists.apply(lambda parts: 0 if not parts else len(parts))

        word_count = word_count.astype("Int64")



        _try_add(

            working,

            feature_name=f"{col}_CharCount",

            values=char_count,

            source=col,

            transformation="Character count (after strip)",

            reason="Provides a simple measure of text length in characters.",

            generated=generated,

            skipped=skipped,

            selected=selected,

        )

        _try_add(

            working,

            feature_name=f"{col}_WordCount",

            values=word_count,

            source=col,

            transformation="Number of whitespace-separated words",

            reason="Provides a simple measure of review/text length in words.",

            generated=generated,

            skipped=skipped,

            selected=selected,

        )



    if len(working) != before_rows:

        raise TextFeatureEngineeringError(

            "Row count changed during text feature engineering."

        )



    return working, {"generated": generated, "skipped": skipped}


