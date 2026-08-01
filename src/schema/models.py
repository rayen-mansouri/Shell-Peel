from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrosswalkRecord:
    crosswalk_record_id: str
    entity_a: str
    entity_b: str
    link_type: str
    evidence: str
    reviewer: str
    rationale: str


@dataclass(frozen=True)
class FusionRecord:
    internal_entity_id: str
    entity_name: str
    known_identifiers: dict[str, Any]
    identifier_confidence: str
    crosswalk_record_id: str
    track: str
    osint_signals: dict[str, Any]
    market_signals: dict[str, Any]
    fused_priority_ranking: str
