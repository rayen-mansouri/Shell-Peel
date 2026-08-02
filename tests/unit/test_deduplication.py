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


def test_dedup_transitive_bridging_behavior_is_explicit():
    """Transitive bridging via connected-components is intentional.

    ring1∩ring2 shares {H1, H2} (meets threshold=2),
    ring2∩ring3 shares {K1, K2} (meets threshold=2),
    ring1∩ring3 shares nothing.

    Design decision: _dedupe_into_rings uses nx.connected_components on
    the pairwise-edge graph.  If ring1-ring2 and ring2-ring3 each cross
    the merge threshold, all three collapse into one component even though
    ring1 and ring3 have no direct pairwise overlap.  This is intentional:
    shared hub accounts between ring1/ring2 and ring2/ring3 constitute
    plausible evidence of one larger laundering network routing through
    common infrastructure.  The test asserts this *collapses to one group*
    so any future change to non-transitive merging breaks this test and
    forces an explicit decision.
    """
    engine = FinancialGraphEngine(min_shared_nodes_to_merge=2)

    t0 = datetime(2024, 1, 1)
    # ring1: H1, H2, A, B  (shares H1,H2 with ring2)
    ring1 = [
        ("H1", "H2", t0, 100.0),
        ("H2", "A", t0, 95.0),
        ("A", "B", t0, 90.0),
        ("B", "H1", t0, 85.0),
    ]
    # ring2: H1, H2, K1, K2  (bridges ring1 and ring3 — shares H1,H2 with ring1, K1,K2 with ring3)
    ring2 = [
        ("H1", "H2", t0, 50.0),
        ("H2", "K1", t0, 48.0),
        ("K1", "K2", t0, 46.0),
        ("K2", "H1", t0, 44.0),
    ]
    # ring3: K1, K2, C, D  (shares K1,K2 with ring2; nothing with ring1)
    ring3 = [
        ("K1", "K2", t0, 30.0),
        ("K2", "C", t0, 28.0),
        ("C", "D", t0, 26.0),
        ("D", "K1", t0, 24.0),
    ]

    merged = engine._dedupe_into_rings([ring1, ring2, ring3])

    # All three transitively connected → one merged group
    assert len(merged) == 1, (
        f"Expected 1 merged ring (transitive bridging), got {len(merged)}: {merged}"
    )

    merged_nodes = set(merged[0])
    # Every node from every ring must appear in the one merged group
    for expected_node in {"H1", "H2", "A", "B", "K1", "K2", "C", "D"}:
        assert expected_node in merged_nodes, f"{expected_node!r} missing from merged ring"

