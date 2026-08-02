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


def _collect_dict_keys_recursive(obj: Any) -> set[str]:
    """Recursively collect all dict keys found at any depth within *obj*.

    Descends into dict values and into list/tuple elements to reach any
    nested dicts. Non-collection scalars terminate the recursion.
    """
    keys: set[str] = set()
    if isinstance(obj, dict):
        keys.update(obj.keys())
        for v in obj.values():
            keys.update(_collect_dict_keys_recursive(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.update(_collect_dict_keys_recursive(item))
    return keys


def validate_screening_language(record: dict[str, Any]) -> None:
    """Validate that the record does not use forbidden screening language.

    Forbidden top-level field names:
    - ``market_risk_score``  → use ``market_screening_score``
    - ``fused_risk_category`` → use ``fused_priority_ranking``

    Forbidden Mode A capability claims (``wash_trading_detected``,
    ``circular_trade_partners``): these keys must not appear *anywhere*
    inside ``market_signals`` or ``osint_signals``, including nested
    sub-dicts at any depth.  The check is scoped to these two sub-dicts
    rather than the entire record because the terms may legitimately appear
    in audit metadata or rationale fields outside the signal blocks.
    """
    if "market_risk_score" in record:
        raise ValueError("Use market_screening_score, not market_risk_score")
    if "fused_risk_category" in record:
        raise ValueError("Use fused_priority_ranking, not fused_risk_category")

    forbidden = {"wash_trading_detected", "circular_trade_partners"}
    for block_name in ("market_signals", "osint_signals"):
        block = record.get(block_name, {}) or {}
        all_keys = _collect_dict_keys_recursive(block)
        present = forbidden & all_keys
        if present:
            raise ValueError(
                f"{block_name} contains claims Mode A cannot support: {sorted(present)}"
            )


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
