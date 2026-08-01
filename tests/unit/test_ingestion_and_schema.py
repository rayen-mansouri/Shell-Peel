from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import discover_amlsim_scenarios, load_crosswalk, validate_scenario_separability
from schema.validators import validate_crosswalk_record, validate_fusion_record


def test_validate_crosswalk_record_requires_audit_fields():
    validate_crosswalk_record(
        {
            "crosswalk_record_id": "cw_0001",
            "entity_a": "ICIJ:1",
            "entity_b": "CIK:2",
            "link_type": "same beneficial owner",
            "evidence": "manual verification",
            "reviewer": "A",
            "rationale": "matched on name",
        }
    )


def test_validate_fusion_record_requires_priority_language():
    validate_fusion_record(
        {
            "internal_entity_id": "ent_1",
            "entity_name": "Example Ltd",
            "known_identifiers": {},
            "identifier_confidence": "manual-matched",
            "crosswalk_record_id": "cw_1",
            "track": "track_1_case_study",
            "osint_signals": {},
            "market_signals": {},
            "fused_priority_ranking": "HIGH",
        }
    )


def test_validate_fusion_record_rejects_unqualified_market_claims():
    try:
        validate_fusion_record(
            {
                "internal_entity_id": "ent_1",
                "entity_name": "Example Ltd",
                "known_identifiers": {},
                "identifier_confidence": "manual-matched",
                "crosswalk_record_id": "cw_1",
                "track": "track_1_case_study",
                "osint_signals": {},
                "market_signals": {
                    "wash_trading_detected": True,
                    "circular_trade_partners": ["A", "B"],
                },
                "fused_priority_ranking": "HIGH",
            }
        )
    except ValueError as exc:
        assert "wash_trading_detected" in str(exc)
        assert "circular_trade_partners" in str(exc)
    else:
        raise AssertionError("Expected validate_fusion_record to reject unsupported market signals")


def test_scenario_discovery_finds_separate_scenario_dirs(tmp_path: Path):
    scenario_a = tmp_path / "scenario_a"
    scenario_b = tmp_path / "scenario_b"
    scenario_a.mkdir()
    scenario_b.mkdir()
    pd.DataFrame([{"x": 1}]).to_csv(scenario_a / "transactions.csv", index=False)
    pd.DataFrame([{"x": 2}]).to_csv(scenario_b / "transactions.csv", index=False)

    scenarios = discover_amlsim_scenarios(tmp_path)
    assert len(scenarios) == 2
    assert validate_scenario_separability(tmp_path) == scenarios


def test_crosswalk_loader_accepts_required_columns(tmp_path: Path):
    path = tmp_path / "crosswalk.csv"
    pd.DataFrame(
        [
            {
                "crosswalk_record_id": "cw_1",
                "entity_a": "ICIJ:1",
                "entity_b": "CIK:2",
                "link_type": "same beneficial owner",
                "evidence": "manual verification",
                "reviewer": "A",
                "rationale": "matched on name",
            }
        ]
    ).to_csv(path, index=False)

    loaded = load_crosswalk(path)
    assert list(loaded.columns) == [
        "crosswalk_record_id",
        "entity_a",
        "entity_b",
        "link_type",
        "evidence",
        "reviewer",
        "rationale",
    ]