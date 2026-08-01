from __future__ import annotations

from financial_graph_engine import is_decaying_ring


def _mk_hops(amounts):
    return [("a", "b", None, amt) for amt in amounts]


def test_growth_case_fails():
    assert is_decaying_ring(_mk_hops([100, 145])) is False


def test_small_amount_degenerate_case_fails():
    assert is_decaying_ring(_mk_hops([10, 59])) is False


def test_gradual_decay_passes():
    assert is_decaying_ring(_mk_hops([100, 97, 95])) is True
