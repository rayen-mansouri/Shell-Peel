from __future__ import annotations

import itertools
import warnings

import networkx as nx
import pandas as pd

from common.logging_utils import get_logger


class FinancialGraphEngine:
    def __init__(
        self,
        max_window_hours: float = 72.0,
        degree_ratio_threshold: float = 0.3,
        max_candidates: int = 200_000,
        min_shared_nodes_to_merge: int = 2,
        max_edges_before_warning: int = 2_000_000,
    ):
        self.graph = nx.MultiDiGraph()
        self.max_window_hours = max_window_hours
        self.degree_ratio_threshold = degree_ratio_threshold
        self.max_candidates = max_candidates
        self.min_shared_nodes_to_merge = min_shared_nodes_to_merge
        self.max_edges_before_warning = max_edges_before_warning
        self._last_run_stats: dict[str, object] = {}
        self.logger = get_logger(self.__class__.__name__)

    def build_graph_from_transactions(self, transactions: pd.DataFrame):
        required = {"sender_id", "receiver_id", "amount", "timestamp"}
        missing = required - set(transactions.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        tx = transactions[transactions["sender_id"] != transactions["receiver_id"]].copy()
        tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")

        n_before = len(tx)
        tx = tx.dropna(subset=["timestamp", "sender_id", "receiver_id", "amount"])
        n_dropped = n_before - len(tx)
        if n_dropped > 0:
            self.logger.warning(
                "Dropped %s of %s transaction rows before graph construction",
                n_dropped,
                n_before,
            )
            warnings.warn(
                f"Dropped {n_dropped} of {n_before} transaction rows "
                f"({100 * n_dropped / n_before:.2f}%) with missing/unparseable "
                "timestamp, sender_id, receiver_id, or amount before graph "
                "construction.",
                stacklevel=2,
            )

        for sender, receiver, amount, ts in tx[["sender_id", "receiver_id", "amount", "timestamp"]].itertuples(
            index=False,
            name=None,
        ):
            self.graph.add_edge(str(sender), str(receiver), amount=float(amount), timestamp=ts)

        if self.graph.number_of_edges() > self.max_edges_before_warning:
            warnings.warn(
                f"Graph has {self.graph.number_of_edges()} edges, exceeding "
                f"{self.max_edges_before_warning}. MultiDiGraph may become memory-bound.",
                stacklevel=2,
            )

    def detect_circular_routing(self, max_cycle_length: int = 4) -> list[list[str]]:
        candidate_rings, truncated = self._collect_temporal_candidate_rings(max_cycle_length=max_cycle_length)

        if truncated:
            warnings.warn(
                f"detect_circular_routing hit max_candidates={self.max_candidates}. "
                "Results are truncated and biased by cycle enumeration order.",
                stacklevel=2,
            )

        return self._dedupe_into_rings(candidate_rings)

    def detect_temporal_rings(self, max_cycle_length: int = 4) -> list[list[tuple[str, str, object, float]]]:
        candidate_rings, truncated = self._collect_temporal_candidate_rings(max_cycle_length=max_cycle_length)

        if truncated:
            warnings.warn(
                f"detect_temporal_rings hit max_candidates={self.max_candidates}. "
                "Results are truncated and biased by cycle enumeration order.",
                stacklevel=2,
            )

        return candidate_rings

    def _collect_temporal_candidate_rings(
        self,
        max_cycle_length: int = 4,
    ) -> tuple[list[list[tuple[str, str, object, float]]], bool]:
        sccs = nx.strongly_connected_components(self.graph)
        nontrivial_scc_nodes = {n for scc in sccs if len(scc) > 1 for n in scc}
        scc_filtered = self.graph.subgraph(nontrivial_scc_nodes)

        balanced_nodes = {
            n
            for n in scc_filtered.nodes
            if abs(scc_filtered.in_degree(n) - scc_filtered.out_degree(n))
            / max(scc_filtered.in_degree(n), scc_filtered.out_degree(n), 1)
            < self.degree_ratio_threshold
        }
        pruned = scc_filtered.subgraph(balanced_nodes)
        simple_topology = nx.DiGraph(pruned)
        raw_cycles = nx.simple_cycles(simple_topology, length_bound=max_cycle_length)

        oversampled = list(itertools.islice(raw_cycles, self.max_candidates + 1))
        truncated = len(oversampled) > self.max_candidates
        to_process = oversampled[: self.max_candidates]

        self._last_run_stats = {
            "candidates_generated_at_least": len(oversampled),
            "candidates_processed": len(to_process),
            "truncated": truncated,
        }

        candidate_rings = []
        for cycle in to_process:
            if len(cycle) < 2:
                continue
            chosen = None
            for offset in range(len(cycle)):
                rotated_cycle = cycle[offset:] + cycle[:offset]
                hop_edges = [(rotated_cycle[i], rotated_cycle[(i + 1) % len(rotated_cycle)]) for i in range(len(rotated_cycle))]
                chosen = self._find_valid_temporal_path(hop_edges)
                if chosen is not None:
                    break
            if chosen is not None:
                candidate_rings.append(chosen)

        self._last_run_stats["accepted_temporal_rings"] = len(candidate_rings)
        return candidate_rings, truncated

    def _find_valid_temporal_path(self, hop_edges, start_after=None):
        if not hop_edges:
            return []

        u, v = hop_edges[0]
        edge_data = self.graph.get_edge_data(u, v)
        if not edge_data:
            return None

        options = sorted(
            (d["timestamp"], d["amount"])
            for d in edge_data.values()
            if start_after is None or d["timestamp"] >= start_after
        )

        for ts, amt in options:
            if start_after is not None and (ts - start_after).total_seconds() > self.max_window_hours * 3600:
                break
            rest = self._find_valid_temporal_path(hop_edges[1:], start_after=ts)
            if rest is not None:
                return [(u, v, ts, amt)] + rest

        return None

    def _dedupe_into_rings(self, candidate_rings: list) -> list[list[str]]:
        """Merge near-duplicate candidate rings into deduplicated laundering rings.

        Two candidate rings are considered the same underlying ring if they share
        at least ``min_shared_nodes_to_merge`` account nodes.  Merging is
        performed via ``nx.connected_components`` on the pairwise-edge graph,
        which means **transitive bridging is intentional**: if ring1 and ring2
        share enough hub accounts to meet the threshold, and ring2 and ring3
        share enough *different* hub accounts to meet the threshold, all three
        collapse into one merged ring even though ring1 and ring3 have no direct
        pairwise overlap.

        **Design rationale:** In an AML context, shared hub accounts are the
        meaningful signal.  If ring1/ring2 co-route through hubs H1/H2 and
        ring2/ring3 co-route through hubs K1/K2, the bridging ring2 constitutes
        evidence that all three cycles belong to one larger laundering network
        routing through common infrastructure.  Collapsing them avoids
        fragmenting what is likely one coordinated operation into multiple
        separate alerts.

        **Consequence to be aware of:** genuinely *distinct* rings that happen to
        share a common intermediary (e.g. a widely-used correspondent bank) could
        also be collapsed.  If ``min_shared_nodes_to_merge`` is set too low this
        can produce spuriously large merged rings.  Tune the threshold and
        cross-check against the topology-matched null baseline to detect collapse
        artefacts.
        """

        ring_graph = nx.Graph()
        node_sets = []

        for i, cyc in enumerate(candidate_rings):
            nodes = {hop[0] for hop in cyc} | {hop[1] for hop in cyc}
            node_sets.append(nodes)
            ring_graph.add_node(i)

        for i in range(len(node_sets)):
            for j in range(i + 1, len(node_sets)):
                shared = node_sets[i] & node_sets[j]
                if len(shared) >= self.min_shared_nodes_to_merge:
                    ring_graph.add_edge(i, j)

        merged = []
        for component in nx.connected_components(ring_graph):
            all_nodes = set()
            for i in component:
                all_nodes |= node_sets[i]
            merged.append(sorted(all_nodes))

        merged.sort(key=lambda r: (len(r), r))
        return merged


def is_decaying_ring(chosen_hops: list, rel_tolerance: float = 0.05, abs_slack: float = 2.0) -> bool:
    amounts = [h[3] for h in chosen_hops]
    for i in range(len(amounts) - 1):
        allowed_increase = min(amounts[i] * rel_tolerance, abs_slack)
        if amounts[i + 1] > amounts[i] + allowed_increase:
            return False
    return True
