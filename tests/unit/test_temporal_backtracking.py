from __future__ import annotations

import pandas as pd

from financial_graph_engine import FinancialGraphEngine


def test_backtracking_finds_valid_path_when_greedy_would_fail():
    engine = FinancialGraphEngine(max_window_hours=5)

    tx = pd.DataFrame(
        [
            {"sender_id": "A", "receiver_id": "B", "amount": 100.0, "timestamp": "2024-01-01T00:00:00"},
            {"sender_id": "A", "receiver_id": "B", "amount": 101.0, "timestamp": "2024-01-01T08:00:00"},
            {"sender_id": "B", "receiver_id": "C", "amount": 99.0, "timestamp": "2024-01-01T09:00:00"},
        ]
    )
    engine.build_graph_from_transactions(tx)

    chosen = engine._find_valid_temporal_path([("A", "B"), ("B", "C")])
    assert chosen is not None
    assert chosen[0][2] == pd.Timestamp("2024-01-01T08:00:00")
    assert chosen[1][2] == pd.Timestamp("2024-01-01T09:00:00")
