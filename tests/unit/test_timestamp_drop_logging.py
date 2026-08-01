from __future__ import annotations

import pandas as pd
import pytest

from financial_graph_engine import FinancialGraphEngine


def test_bad_timestamp_rows_are_dropped_and_warned():
    engine = FinancialGraphEngine()
    tx = pd.DataFrame(
        [
            {"sender_id": "A", "receiver_id": "B", "amount": 100.0, "timestamp": "2024-01-01T00:00:00"},
            {"sender_id": "B", "receiver_id": "C", "amount": 90.0, "timestamp": "bad_ts"},
            {"sender_id": "C", "receiver_id": "D", "amount": 80.0, "timestamp": None},
        ]
    )

    with pytest.warns(UserWarning, match="Dropped 2 of 3"):
        engine.build_graph_from_transactions(tx)

    assert engine.graph.number_of_edges() == 1
