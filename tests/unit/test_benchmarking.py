from __future__ import annotations

import networkx as nx

from common import bootstrap_ci, performance_profile_row, ring_metrics, sample_topology_matched_null_rings
from financial_graph_engine import FinancialGraphEngine


def test_ring_metrics_counts_exact_matches_and_account_recall():
    detected = [["A", "B", "C"], ["X", "Y"]]
    truth = [["A", "B", "C"], ["P", "Q"]]

    metrics = ring_metrics(detected, truth)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["account_recall"] == 3 / 5


def test_bootstrap_ci_returns_finite_interval():
    point, lo, hi = bootstrap_ci([1, 2, 3, 4], lambda values: sum(values) / len(values), n_resamples=200, seed=1)
    assert lo <= point <= hi
    assert lo <= hi


def test_sample_topology_matched_null_rings_preserves_sizes():
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "A")
    graph.add_edge("A", "D")

    null_rings = sample_topology_matched_null_rings(graph, [["A", "B", "C"], ["A", "D"]], seed=7)
    assert [len(ring) for ring in null_rings] == [3, 2]


def test_performance_profile_row_uses_engine_statistics():
    engine = FinancialGraphEngine()
    tx = __import__("pandas").DataFrame(
        [
            {"sender_id": "A", "receiver_id": "B", "amount": 10.0, "timestamp": "2024-01-01T00:00:00"},
            {"sender_id": "B", "receiver_id": "C", "amount": 9.0, "timestamp": "2024-01-01T01:00:00"},
            {"sender_id": "C", "receiver_id": "A", "amount": 8.0, "timestamp": "2024-01-01T02:00:00"},
        ]
    )
    engine.build_graph_from_transactions(tx)
    engine.detect_circular_routing()

    row = performance_profile_row(engine, "toy-scenario", 0.25, peak_memory_bytes=1024 * 1024)
    assert row["scenario"] == "toy-scenario"
    assert row["scenario_edges"] == 3
    assert row["accepted_temporal_rings"] >= 1
    assert row["runtime_seconds"] == 0.25
    assert row["peak_memory_mb"] == 1.0
