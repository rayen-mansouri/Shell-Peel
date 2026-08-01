from __future__ import annotations

from datetime import datetime

from financial_graph_engine import FinancialGraphEngine


def test_single_shared_hub_does_not_merge_when_threshold_is_two():
    engine = FinancialGraphEngine(min_shared_nodes_to_merge=2)

    t0 = datetime(2024, 1, 1)
    ring_1 = [
        ("H", "A", t0, 10.0),
        ("A", "B", t0, 9.5),
        ("B", "H", t0, 9.0),
    ]
    ring_2 = [
        ("H", "C", t0, 20.0),
        ("C", "D", t0, 19.0),
        ("D", "H", t0, 18.0),
    ]

    merged = engine._dedupe_into_rings([ring_1, ring_2])
    assert len(merged) == 2
