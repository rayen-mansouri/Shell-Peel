from __future__ import annotations

from typing import Any


def validate_crosswalk_record(record: dict[str, Any]) -> None:
    required = {"crosswalk_record_id", "entity_a", "entity_b", "link_type", "evidence", "reviewer", "rationale"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"Missing crosswalk fields: {sorted(missing)}")


def validate_track(record: dict[str, Any]) -> None:
    track = record.get("track")
    allowed = {"track_1_case_study", "track_2_synthetic_benchmark"}
    if track not in allowed:
        raise ValueError(f"Invalid track: {track}. Expected one of {sorted(allowed)}")


def validate_screening_language(record: dict[str, Any]) -> None:
    if "market_risk_score" in record:
        raise ValueError("Use market_screening_score, not market_risk_score")
    if "fused_risk_category" in record:
        raise ValueError("Use fused_priority_ranking, not fused_risk_category")

    market_signals = record.get("market_signals", {}) or {}
    forbidden = {"wash_trading_detected", "circular_trade_partners"}
    present = forbidden & set(market_signals)
    if present:
        raise ValueError(f"market_signals contains claims Mode A cannot support: {sorted(present)}")


def validate_fusion_record(record: dict[str, Any]) -> None:
    required = {
        "internal_entity_id",
        "entity_name",
        "known_identifiers",
        "identifier_confidence",
        "crosswalk_record_id",
        "track",
        "osint_signals",
        "market_signals",
        "fused_priority_ranking",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"Missing fusion fields: {sorted(missing)}")
    validate_track(record)
    validate_screening_language(record)
