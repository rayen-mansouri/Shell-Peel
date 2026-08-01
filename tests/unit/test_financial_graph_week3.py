from __future__ import annotations

import pandas as pd

from financial_graph_engine import FinancialGraphEngine, is_decaying_ring


def test_detect_temporal_rings_exposes_raw_hop_sequences():
    engine = FinancialGraphEngine()
    tx = pd.DataFrame(
        [
            {"sender_id": "A", "receiver_id": "B", "amount": 10.0, "timestamp": "2024-01-01T00:00:00"},
            {"sender_id": "B", "receiver_id": "C", "amount": 9.0, "timestamp": "2024-01-01T01:00:00"},
            {"sender_id": "C", "receiver_id": "A", "amount": 8.0, "timestamp": "2024-01-01T02:00:00"},
        ]
    )
    engine.build_graph_from_transactions(tx)

    raw_temporal_rings = engine.detect_temporal_rings()
    assert raw_temporal_rings
    assert all(isinstance(hop, tuple) and len(hop) == 4 for hop in raw_temporal_rings[0])
    assert is_decaying_ring(raw_temporal_rings[0]) is True


def test_detect_temporal_rings_handles_rotated_cycle_order():
    engine = FinancialGraphEngine()
    tx = pd.DataFrame(
        [
            {"sender_id": "A", "receiver_id": "B", "amount": 10.0, "timestamp": "2024-01-01T00:00:00"},
            {"sender_id": "B", "receiver_id": "C", "amount": 9.0, "timestamp": "2024-01-01T01:00:00"},
            {"sender_id": "C", "receiver_id": "A", "amount": 8.0, "timestamp": "2024-01-01T02:00:00"},
        ]
    )
    engine.build_graph_from_transactions(tx)

    rotated_rings = engine._collect_temporal_candidate_rings(max_cycle_length=4)[0]
    assert rotated_rings
    assert rotated_rings[0][0][0] == "A"


def test_detect_circular_routing_returns_deduped_node_rings():
    engine = FinancialGraphEngine()
    tx = pd.DataFrame(
        [
            {"sender_id": "A", "receiver_id": "B", "amount": 10.0, "timestamp": "2024-01-01T00:00:00"},
            {"sender_id": "B", "receiver_id": "C", "amount": 9.0, "timestamp": "2024-01-01T01:00:00"},
            {"sender_id": "C", "receiver_id": "A", "amount": 8.0, "timestamp": "2024-01-01T02:00:00"},
        ]
    )
    engine.build_graph_from_transactions(tx)

    rings = engine.detect_circular_routing()
    assert rings == [["A", "B", "C"]]
