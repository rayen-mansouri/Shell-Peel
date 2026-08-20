from __future__ import annotations

from collections.abc import Callable, Sequence
import itertools
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


def _normalize_ring(ring: Any) -> frozenset[str]:
    items = list(ring or [])
    if not items:
        return frozenset()

    first_item = items[0]
    if isinstance(first_item, (tuple, list)) and len(first_item) >= 2:
        nodes = {str(hop[0]) for hop in items} | {str(hop[1]) for hop in items}
        return frozenset(nodes)

    return frozenset(str(node) for node in items)


def ring_metrics(detected_rings: Sequence[Any], truth_rings: Sequence[Any]) -> dict[str, float]:
    detected_norm = [_normalize_ring(ring) for ring in detected_rings if _normalize_ring(ring)]
    truth_norm = {_normalize_ring(ring) for ring in truth_rings if _normalize_ring(ring)}

    tp_count = sum(1 for ring in detected_norm if ring in truth_norm)
    unique_matches = {ring for ring in detected_norm if ring in truth_norm}

    precision = tp_count / len(detected_norm) if detected_norm else 0.0
    recall = len(unique_matches) / len(truth_norm) if truth_norm else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    truth_accounts = set().union(*truth_norm) if truth_norm else set()
    recovered_accounts = set().union(*unique_matches) if unique_matches else set()
    account_recall = len(recovered_accounts) / len(truth_accounts) if truth_accounts else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "account_recall": float(account_recall),
        "detected_rings": float(len(detected_norm)),
        "truth_rings": float(len(truth_norm)),
    }


def aggregate_scenario_results(scenario_results: Sequence[tuple[Sequence[Any], Sequence[Any]]]) -> dict[str, float]:
    all_detected: list[Any] = []
    all_truth: list[Any] = []
    for detected_rings, truth_rings in scenario_results:
        all_detected.extend(detected_rings)
        all_truth.extend(truth_rings)
    return ring_metrics(all_detected, all_truth)


def bootstrap_ci(
    samples: Sequence[Any],
    metric_fn: Callable[[Sequence[Any]], float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    point_estimate = float(metric_fn(samples))
    rng = np.random.default_rng(seed)
    n_items = len(samples)
    boot_estimates = []

    for _ in range(n_resamples):
        resample = [samples[index] for index in rng.integers(0, n_items, size=n_items)]
        boot_estimates.append(float(metric_fn(resample)))

    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_estimates, [100 * alpha, 100 * (1 - alpha)])
    return point_estimate, float(lo), float(hi)


def graph_stage_counts(graph: nx.MultiDiGraph, degree_ratio_threshold: float) -> dict[str, int]:
    sccs = nx.strongly_connected_components(graph)
    nontrivial_scc_nodes = {node for component in sccs if len(component) > 1 for node in component}
    scc_filtered = graph.subgraph(nontrivial_scc_nodes)

    balanced_nodes = {
        node
        for node in scc_filtered.nodes
        if abs(scc_filtered.in_degree(node) - scc_filtered.out_degree(node))
        / max(scc_filtered.in_degree(node), scc_filtered.out_degree(node), 1)
        < degree_ratio_threshold
    }

    return {
        "scenario_edges": graph.number_of_edges(),
        "scenario_nodes": graph.number_of_nodes(),
        "nodes_after_scc_filter": len(nontrivial_scc_nodes),
        "nodes_after_degree_prune": len(balanced_nodes),
    }


def performance_profile_row(
    engine: Any,
    scenario_name: str,
    runtime_seconds: float,
    peak_memory_bytes: int | None = None,
) -> dict[str, Any]:
    stage_counts = graph_stage_counts(engine.graph, engine.degree_ratio_threshold)
    stats = getattr(engine, "_last_run_stats", {}) or {}

    return {
        "scenario": scenario_name,
        **stage_counts,
        "candidates_generated_at_least": int(stats.get("candidates_generated_at_least", 0)),
        "candidates_processed": int(stats.get("candidates_processed", 0)),
        "accepted_temporal_rings": int(stats.get("accepted_temporal_rings", 0)),
        "truncated": bool(stats.get("truncated", False)),
        "runtime_seconds": float(runtime_seconds),
        "peak_memory_mb": float(peak_memory_bytes / (1024 * 1024)) if peak_memory_bytes is not None else np.nan,
    }


def sample_topology_matched_null_rings(
    graph: nx.MultiDiGraph,
    truth_rings: Sequence[Any],
    max_cycle_length: int = 4,
    max_pool_size: int = 50_000,
    seed: int = 42,
) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    simple_topology = nx.DiGraph(graph)
    cycle_pool = list(itertools.islice(nx.simple_cycles(simple_topology, length_bound=max_cycle_length), max_pool_size))
    truth_norm = {_normalize_ring(ring) for ring in truth_rings if _normalize_ring(ring)}
    pool_by_size: dict[int, list[list[str]]] = {}

    for cycle in cycle_pool:
        norm_cycle = _normalize_ring(cycle)
        if norm_cycle in truth_norm:
            continue
        pool_by_size.setdefault(len(norm_cycle), []).append(cycle)

    null_rings: list[list[str]] = []
    for ring in truth_rings:
        ring_size = max(len(_normalize_ring(ring)), 2)
        candidates = pool_by_size.get(ring_size, [])
        if not candidates:
            continue

        chosen = candidates[rng.integers(0, len(candidates))]
        null_rings.append([str(node) for node in chosen])

    return null_rings