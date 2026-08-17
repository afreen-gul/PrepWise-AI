"""Phase 6.13–6.15 — transparent feature decision engine."""

from __future__ import annotations

from typing import Any

from app.schemas.feature_selection import (
    CorrelationPairRow,
    FeatureDecisionRow,
    FeatureQualityRow,
    FeatureSelectionSummary,
    TargetScoreRow,
    VIFRow,
)


def build_feature_decisions(
    *,
    quality_rows: list[FeatureQualityRow],
    correlation_pairs: list[CorrelationPairRow],
    vif_rows: list[VIFRow],
    target_scores: list[TargetScoreRow],
    target_column: str | None,
) -> tuple[list[FeatureDecisionRow], FeatureSelectionSummary, list[str]]:
    """Combine evidence into KEEP / REVIEW / REMOVE decisions."""
    corr_note: dict[str, list[str]] = {}
    for pair in correlation_pairs:
        msg = (
            f"High correlation ({pair.correlation}) with "
            f"{pair.feature_b if pair.feature_a else ''}".strip()
        )
        # clearer per feature
        corr_note.setdefault(pair.feature_a, []).append(
            f"|corr|={abs(pair.correlation):.2f} with {pair.feature_b}"
        )
        corr_note.setdefault(pair.feature_b, []).append(
            f"|corr|={abs(pair.correlation):.2f} with {pair.feature_a}"
        )

    vif_by = {r.feature: r for r in vif_rows}
    mi_by = {r.feature: r for r in target_scores}
    explanations: list[str] = []

    decisions: list[FeatureDecisionRow] = []
    for q in quality_rows:
        name = q.feature
        evidence: list[str] = []
        methods: list[str] = []
        is_target = target_column is not None and name == target_column

        if is_target:
            decisions.append(
                FeatureDecisionRow(
                    feature=name,
                    feature_type=q.semantic_type,
                    status="TARGET",
                    missing_pct=q.missing_pct,
                    unique_pct=q.unique_pct,
                    correlation="N/A",
                    vif="N/A",
                    target_score="N/A",
                    decision="KEEP",
                    reason="Target / label column — always preserved.",
                    evidence=["Protected as target"],
                    methods=["target_protection"],
                    is_target=True,
                    is_generated=q.is_generated,
                    source_feature=q.source_feature,
                    transformation=q.transformation,
                )
            )
            continue

        # Strong REMOVE evidence
        if q.is_constant:
            decisions.append(
                FeatureDecisionRow(
                    feature=name,
                    feature_type=q.semantic_type,
                    status="CONSTANT",
                    missing_pct=q.missing_pct,
                    unique_pct=q.unique_pct,
                    correlation="N/A",
                    vif="N/A",
                    target_score=_fmt_mi(mi_by.get(name)),
                    decision="REMOVE",
                    reason="Constant feature — provides no information for modeling.",
                    evidence=["unique_count <= 1"],
                    methods=["feature_quality"],
                    is_generated=q.is_generated,
                    source_feature=q.source_feature,
                    transformation=q.transformation,
                )
            )
            continue

        if q.is_exact_duplicate and q.duplicate_of:
            decisions.append(
                FeatureDecisionRow(
                    feature=name,
                    feature_type=q.semantic_type,
                    status="EXACT_DUPLICATE",
                    missing_pct=q.missing_pct,
                    unique_pct=q.unique_pct,
                    correlation="N/A",
                    vif="N/A",
                    target_score=_fmt_mi(mi_by.get(name)),
                    decision="REMOVE",
                    reason=(
                        f"Exact duplicate of `{q.duplicate_of}` — values match "
                        "row-for-row. Prefer keeping the original feature."
                    ),
                    evidence=[f"duplicate_of={q.duplicate_of}"],
                    methods=["exact_duplicate_detection"],
                    is_generated=q.is_generated,
                    source_feature=q.source_feature,
                    transformation=q.transformation,
                )
            )
            continue

        if q.is_identifier:
            decisions.append(
                FeatureDecisionRow(
                    feature=name,
                    feature_type=q.semantic_type,
                    status="IDENTIFIER",
                    missing_pct=q.missing_pct,
                    unique_pct=q.unique_pct,
                    correlation="N/A",
                    vif="N/A",
                    target_score=_fmt_mi(mi_by.get(name)),
                    decision="REMOVE",
                    reason=(
                        "Identifier-like feature with extremely high uniqueness "
                        "and/or identifier naming — usually harmful for ML."
                    ),
                    evidence=q.quality_flags,
                    methods=["identifier_detection"],
                    is_generated=q.is_generated,
                    source_feature=q.source_feature,
                    transformation=q.transformation,
                )
            )
            continue

        # Accumulate REVIEW signals
        review_reasons: list[str] = []
        status_bits: list[str] = []

        if q.is_near_constant:
            review_reasons.append(
                "Near-constant: one value dominates almost all rows."
            )
            status_bits.append("NEAR_CONSTANT")
            methods.append("feature_quality")
            evidence.append("near_constant")

        if "high_missing" in q.quality_flags:
            review_reasons.append(
                f"High missingness ({q.missing_pct:.1f}%) — review usefulness."
            )
            status_bits.append("HIGH_MISSING")
            methods.append("feature_quality")
            evidence.append(f"missing_pct={q.missing_pct}")
        elif "moderate_missing" in q.quality_flags:
            review_reasons.append(
                f"Moderate missingness ({q.missing_pct:.1f}%)."
            )
            status_bits.append("MODERATE_MISSING")
            methods.append("feature_quality")
            evidence.append(f"missing_pct={q.missing_pct}")

        if "high_cardinality" in q.quality_flags:
            review_reasons.append(
                "High-cardinality categorical — may need encoding care; "
                "not removed automatically."
            )
            status_bits.append("HIGH_CARDINALITY")
            methods.append("feature_quality")
            evidence.append(f"unique_pct={q.unique_pct}")

        corr_msgs = corr_note.get(name) or []
        corr_display = "N/A"
        if corr_msgs:
            corr_display = "; ".join(corr_msgs[:2])
            review_reasons.append(
                "Highly correlated with another numerical feature — "
                "overlapping information may add limited value."
            )
            status_bits.append("HIGH_CORRELATION")
            methods.append("correlation_analysis")
            evidence.extend(corr_msgs[:2])
            explanations.append(
                f"`{name}` shares overlapping information with related features "
                f"({corr_msgs[0]}). Keeping both may provide limited additional "
                "value and can increase redundancy."
            )

        vif_row = vif_by.get(name)
        vif_display = "N/A"
        if vif_row and vif_row.vif is not None:
            vif_display = f"{vif_row.vif} ({vif_row.status})"
            if vif_row.status in {"HIGH", "REVIEW"}:
                review_reasons.append(
                    "Several numerical features contain overlapping information, "
                    "which can make some models less stable (elevated VIF)."
                )
                status_bits.append(f"VIF_{vif_row.status}")
                methods.append("vif_analysis")
                evidence.append(f"vif={vif_row.vif}")

        mi_row = mi_by.get(name)
        mi_display = _fmt_mi(mi_row)
        useful_mi = bool(mi_row and mi_row.mi_score is not None and mi_row.mi_score >= 0.02)

        if review_reasons:
            # Conflicting: strong MI + redundancy → still REVIEW
            reason = " ".join(review_reasons)
            if useful_mi:
                reason += (
                    " Target relationship looks useful, but redundancy/quality "
                    "signals warrant REVIEW rather than automatic removal."
                )
            decisions.append(
                FeatureDecisionRow(
                    feature=name,
                    feature_type=q.semantic_type,
                    status=", ".join(status_bits) or "REVIEW",
                    missing_pct=q.missing_pct,
                    unique_pct=q.unique_pct,
                    correlation=corr_display,
                    vif=vif_display,
                    target_score=mi_display,
                    decision="REVIEW",
                    reason=reason,
                    evidence=evidence,
                    methods=sorted(set(methods)),
                    is_generated=q.is_generated,
                    source_feature=q.source_feature,
                    transformation=q.transformation,
                )
            )
            continue

        # KEEP — no significant structural problem
        keep_reason = "No significant structural or redundancy issues detected."
        if useful_mi:
            keep_reason = (
                "Acceptable quality with a useful statistical relationship "
                "to the target."
            )
            evidence.append(f"mi={mi_row.mi_score}")
            methods.append("mutual_information")
        if q.is_generated:
            keep_reason += (
                f" Generated in Phase 5 from `{q.source_feature}` "
                f"({q.transformation})."
            )
            methods.append("phase5_metadata")

        decisions.append(
            FeatureDecisionRow(
                feature=name,
                feature_type=q.semantic_type,
                status="GOOD",
                missing_pct=q.missing_pct,
                unique_pct=q.unique_pct,
                correlation=corr_display,
                vif=vif_display,
                target_score=mi_display,
                decision="KEEP",
                reason=keep_reason,
                evidence=evidence or ["no_strong_negative_signals"],
                methods=sorted(set(methods)) or ["feature_quality"],
                is_generated=q.is_generated,
                source_feature=q.source_feature,
                transformation=q.transformation,
            )
        )

    keep_n = sum(1 for d in decisions if d.decision == "KEEP")
    review_n = sum(1 for d in decisions if d.decision == "REVIEW")
    remove_n = sum(1 for d in decisions if d.decision == "REMOVE")
    summary = FeatureSelectionSummary(
        total_features=len(decisions),
        keep=keep_n,
        review=review_n,
        remove=remove_n,
        target_column=target_column,
        target_task=(
            target_scores[0].target_type if target_scores else None
        ),
        target_aware_applied=bool(target_scores),
    )

    # Deduplicate explanations
    uniq_expl: list[str] = []
    for e in explanations:
        if e not in uniq_expl:
            uniq_expl.append(e)
    return decisions, summary, uniq_expl[:20]


def _fmt_mi(row: TargetScoreRow | None) -> str:
    if row is None or row.mi_score is None:
        return "N/A"
    return f"{row.mi_score:.4f} (rank {row.rank})"


def columns_to_drop_for_recommended_selection(
    decisions: list[FeatureDecisionRow],
    *,
    also_remove: list[str] | None = None,
    force_keep: list[str] | None = None,
    target_column: str | None = None,
) -> list[str]:
    """
    Safe default: drop REMOVE only.
    REVIEW kept unless listed in also_remove.
    Target never dropped.
    """
    also_remove = set(also_remove or [])
    force_keep = set(force_keep or [])
    drop: list[str] = []
    for d in decisions:
        if target_column and d.feature == target_column:
            continue
        if d.feature in force_keep:
            continue
        if d.decision == "REMOVE":
            drop.append(d.feature)
        elif d.feature in also_remove:
            drop.append(d.feature)
    return drop
