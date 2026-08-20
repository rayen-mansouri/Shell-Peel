"""Week 3 quantitative validation — leak-free timestamp split, Steps 0–7."""
from __future__ import annotations

import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from common.benchmarking import (
    bootstrap_ci,
    performance_profile_row,
    ring_metrics,
    sample_topology_matched_null_rings,
)
from common.ingestion import validate_scenario_separability
from financial_graph_engine.engine import FinancialGraphEngine, is_decaying_ring

ROOT = Path(__file__).resolve().parent.parent
TRANS_FILE = ROOT / "HI-Small_Trans.csv"
PATTERNS_FILE = ROOT / "HI-Small_Patterns.txt"
CUTOFF = pd.to_datetime("2022/09/08 00:00:00")
CHUNK_SIZE = 500_000


def parse_patterns(file_path: Path) -> list[dict]:
    """Parse CYCLE blocks; return dicts with accounts, min_ts, max_ts."""
    truth_rings: list[dict] = []
    with open(file_path, encoding="utf-8") as f:
        in_cycle = False
        current_ring: list[str] = []
        current_timestamps: list[pd.Timestamp] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT - CYCLE"):
                in_cycle = True
                current_ring, current_timestamps = [], []
            elif line.startswith("END LAUNDERING ATTEMPT - CYCLE"):
                in_cycle = False
                if current_ring:
                    truth_rings.append(
                        {
                            "accounts": frozenset(current_ring),
                            "min_ts": min(current_timestamps),
                            "max_ts": max(current_timestamps),
                        }
                    )
            elif in_cycle:
                parts = line.split(",")
                if len(parts) >= 5:
                    current_timestamps.append(pd.to_datetime(parts[0].strip()))
                    current_ring.append(parts[2].strip())
                    current_ring.append(parts[4].strip())
    return truth_rings


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    dev_chunks: list[pd.DataFrame] = []
    test_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(TRANS_FILE, chunksize=CHUNK_SIZE):
        cols = list(chunk.columns)
        acct_indices = [i for i, c in enumerate(cols) if "Account" in c]
        if len(acct_indices) >= 2:
            cols[acct_indices[0]] = "sender_id"
            cols[acct_indices[1]] = "receiver_id"
        chunk.columns = cols
        chunk["timestamp"] = pd.to_datetime(chunk["Timestamp"])
        # Amount Paid = outbound amount from sender; matches edge semantics in engine.
        chunk["amount"] = chunk["Amount Paid"]
        needed = chunk[["sender_id", "receiver_id", "amount", "timestamp"]]
        dev_chunks.append(needed[needed["timestamp"] < CUTOFF])
        test_chunks.append(needed[needed["timestamp"] >= CUTOFF])
    return pd.concat(dev_chunks, ignore_index=True), pd.concat(test_chunks, ignore_index=True)


def ring_sizes(rings: list) -> tuple[float, float, float]:
    sizes = [len(set(r) if not isinstance(r, frozenset) else r) for r in rings]
    if not sizes:
        return float("nan"), float("nan"), float("nan")
    return float(min(sizes)), float(np.median(sizes)), float(max(sizes))


def pipeline_rings(
    engine: FinancialGraphEngine,
    *,
    apply_decay: bool = True,
    rel_tolerance: float = 0.05,
    abs_slack: float = 2.0,
    dedupe: bool = True,
) -> list:
    raw = engine.detect_temporal_rings()
    if apply_decay:
        raw = [r for r in raw if is_decaying_ring(r, rel_tolerance=rel_tolerance, abs_slack=abs_slack)]
    if dedupe:
        return engine._dedupe_into_rings(raw)
    node_rings = []
    for hop_ring in raw:
        nodes = {hop[0] for hop in hop_ring} | {hop[1] for hop in hop_ring}
        node_rings.append(sorted(nodes))
    return node_rings


def ring_level_bootstrap_ci(
    detected_rings: list,
    truth_rings: list,
    n_resamples: int = 500,
) -> dict[str, tuple[float, float, float]]:
    """Resample detected list, hold truth fixed (recall upper bound <= point)."""
    if not detected_rings:
        zero = (0.0, 0.0, 0.0)
        return {"precision": zero, "recall": zero, "f1": zero, "account_recall": zero}

    def prec_fn(resampled_det: list) -> float:
        return ring_metrics(resampled_det, truth_rings)["precision"]

    def rec_fn(resampled_det: list) -> float:
        return ring_metrics(resampled_det, truth_rings)["recall"]

    def f1_fn(resampled_det: list) -> float:
        return ring_metrics(resampled_det, truth_rings)["f1"]

    def acct_rec_fn(resampled_det: list) -> float:
        return ring_metrics(resampled_det, truth_rings)["account_recall"]

    return {
        "precision": bootstrap_ci(detected_rings, prec_fn, n_resamples=n_resamples, seed=42),
        "recall": bootstrap_ci(detected_rings, rec_fn, n_resamples=n_resamples, seed=42),
        "f1": bootstrap_ci(detected_rings, f1_fn, n_resamples=n_resamples, seed=42),
        "account_recall": bootstrap_ci(detected_rings, acct_rec_fn, n_resamples=n_resamples, seed=42),
    }


def fmt_ci(triple: tuple[float, float, float]) -> str:
    pt, lo, hi = triple
    return f"{pt:.4f} [{lo:.4f}, {hi:.4f}]"


def calibrate_decay(engine: FinancialGraphEngine, truth_rings_dev: list[list[str]]) -> tuple[float, float, dict]:
    truth_nodes: set[str] = set()
    for ring in truth_rings_dev:
        truth_nodes.update(ring)
    amounts: list[float] = []
    for u, v, data in engine.graph.edges(data=True):
        if u in truth_nodes and v in truth_nodes:
            amounts.append(float(data["amount"]))
    if not amounts:
        return 0.05, 2.0, {"min": 0, "median": 0, "max": 0, "n": 0}
    arr = np.array(amounts)
    stats = {"min": float(np.min(arr)), "median": float(np.median(arr)), "max": float(np.max(arr)), "n": len(arr)}
    rel_tolerance = 0.05
    abs_slack = max(2.0, float(np.median(arr) * 0.01))
    return rel_tolerance, abs_slack, stats


def fanin_fanout_heavy_rings(graph, truth_rings_dev: list[list[str]], threshold: float = 0.3) -> list[frozenset]:
    """Rings where any member has in/out degree imbalance above threshold."""
    simple = graph.to_directed() if hasattr(graph, "to_directed") else graph
    heavy: list[frozenset] = []
    for ring in truth_rings_dev:
        nodes = list(ring)
        imbalanced = False
        for node in nodes:
            indeg = simple.in_degree(node) if node in simple else 0
            outdeg = simple.out_degree(node) if node in simple else 0
            denom = max(indeg, outdeg, 1)
            if abs(indeg - outdeg) / denom >= threshold:
                imbalanced = True
                break
        if imbalanced:
            heavy.append(frozenset(ring))
    return heavy


def run_step0() -> tuple[pd.DataFrame, pd.DataFrame, list[list[str]], list[list[str]], dict]:
    print("=" * 72)
    print("STEP 0 - Data inventory & leak-free split")
    print("=" * 72)

    with open(TRANS_FILE, encoding="utf-8") as f:
        print("0a TRANS header:", f.readline().strip())
        print("0a TRANS row1:  ", f.readline().strip()[:120])

    with open(PATTERNS_FILE, encoding="utf-8") as f:
        in_cycle = False
        for line in f:
            s = line.strip()
            if s.startswith("BEGIN LAUNDERING ATTEMPT - CYCLE"):
                in_cycle = True
                continue
            if in_cycle and s and not s.startswith("END"):
                print("0a CYCLE raw line:", repr(s[:200]))
                parts = s.split(",")
                print("0a CYCLE parts[0:6]:", [p.strip() for p in parts[:6]])
                break

    all_rings = parse_patterns(PATTERNS_FILE)
    print(f"0c Total CYCLE truth rings parsed: {len(all_rings)}")

    dev_meta = [r for r in all_rings if r["max_ts"] < CUTOFF]
    test_meta = [r for r in all_rings if r["min_ts"] >= CUTOFF]
    straddle = [r for r in all_rings if r["min_ts"] < CUTOFF and r["max_ts"] >= CUTOFF]

    truth_rings_dev = [list(r["accounts"]) for r in dev_meta]
    truth_rings_test = [list(r["accounts"]) for r in test_meta]

    dev_sets = {r["accounts"] for r in dev_meta}
    test_sets = {r["accounts"] for r in test_meta}
    overlap = dev_sets & test_sets

    print(f"0c Cutoff: {CUTOFF}")
    print(f"0c Dev truth rings (max_ts < cutoff): {len(truth_rings_dev)}")
    print(f"0c Test truth rings (min_ts >= cutoff): {len(truth_rings_test)}")
    print(f"0c Straddling rings dropped from BOTH: {len(straddle)}")
    print(f"0c LEAKAGE CHECK - dev/test frozenset overlap: {len(overlap)} sets")
    if overlap:
        print("  OVERLAP (unexpected):", [list(o) for o in list(overlap)[:3]])
    else:
        print("  PASS: overlap is empty.")

    print("0b Loading transactions (chunked, no subsampling)...")
    print("0b Amount column: 'Amount Paid' — outbound payment from sender account, matches engine edge amount.")
    dev_df, test_df = load_splits()
    print(f"0b Dev rows: {len(dev_df):,} | Test rows: {len(test_df):,}")

    print("0d Scenario separability check...")
    try:
        validate_scenario_separability(ROOT)
        print("  Unexpected: found separable scenarios.")
    except ValueError as exc:
        print(f"  PASS: {exc}")
    print("0d Split method: time-based holdout at 2022-09-08 00:00:00 (test held out until Step 6).")

    meta = {
        "overlap_count": len(overlap),
        "straddle_count": len(straddle),
        "n_dev_rings": len(truth_rings_dev),
        "n_test_rings": len(truth_rings_test),
    }
    return dev_df, test_df, truth_rings_dev, truth_rings_test, meta


def main() -> None:
    warnings.filterwarnings("default")
    dev_df, test_df, truth_rings_dev, truth_rings_test, meta = run_step0()

    if meta["overlap_count"] != 0:
        print("BLOCKER: leakage check failed. Stopping.")
        return

    # --- Step 1 ---
    print("\n" + "=" * 72)
    print("STEP 1 - Dev graph & pipeline")
    print("=" * 72)
    tracemalloc.start()
    t0 = time.perf_counter()
    engine = FinancialGraphEngine()
    n_before = len(dev_df)
    engine.build_graph_from_transactions(dev_df)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats = engine._last_run_stats or {}
    detected = engine.detect_circular_routing()
    print(f"Rows fed to engine: {n_before:,}")
    print(f"Graph nodes: {engine.graph.number_of_nodes():,} | edges: {engine.graph.number_of_edges():,}")
    print(f"detect_circular_routing returned {len(detected)} deduped rings")
    print(f"_last_run_stats after detection: {engine._last_run_stats}")
    truncated = bool(engine._last_run_stats.get("truncated", False))
    print(f"max_candidates truncation fired: {truncated}")

    # --- Step 2 ---
    print("\n" + "=" * 72)
    print("STEP 2 - Decay calibration (dev truth amounts)")
    print("=" * 72)
    rel_tol, abs_slack, amt_stats = calibrate_decay(engine, truth_rings_dev)
    print(f"Truth-edge amount distribution (n={amt_stats['n']}): "
          f"min={amt_stats['min']:.2f}, median={amt_stats['median']:.2f}, max={amt_stats['max']:.2f}")
    print(f"Chosen rel_tolerance={rel_tol}, abs_slack={abs_slack:.2f}")
    print("Rationale: 5% relative cap on hop-to-hop amount growth; abs_slack = max($2, 1% of median truth amount).")

    base = {
        "max_window_hours": 72.0,
        "degree_ratio_threshold": 0.3,
        "min_shared_nodes_to_merge": 2,
        "rel_tolerance": rel_tol,
        "abs_slack": abs_slack,
    }

    def apply_base(eng: FinancialGraphEngine) -> None:
        eng.max_window_hours = base["max_window_hours"]
        eng.degree_ratio_threshold = base["degree_ratio_threshold"]
        eng.min_shared_nodes_to_merge = base["min_shared_nodes_to_merge"]

    apply_base(engine)

    # --- Step 3 ---
    print("\n" + "=" * 72)
    print("STEP 3 - Hyperparameter sweep (dev, one-at-a-time)")
    print("=" * 72)
    sweep_rows: list[dict] = []

    def sweep_row(param: str, value, eng: FinancialGraphEngine) -> dict:
        rings = pipeline_rings(
            eng,
            rel_tolerance=base["rel_tolerance"],
            abs_slack=base["abs_slack"],
        )
        m = ring_metrics(rings, truth_rings_dev)
        return {
            "parameter": param,
            "value": value,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "detected": len(rings),
        }

    apply_base(engine)
    sweep_rows.append({**sweep_row("baseline", "defaults", engine)})

    for w in [24, 72, 168]:
        apply_base(engine)
        engine.max_window_hours = float(w)
        sweep_rows.append(sweep_row("max_window_hours", w, engine))

    for factor, label in [(0.5, "-50%"), (1.0, "calibrated"), (1.5, "+50%")]:
        apply_base(engine)
        rt = base["rel_tolerance"] * factor
        ab = base["abs_slack"] * factor
        rings = pipeline_rings(engine, rel_tolerance=rt, abs_slack=ab)
        m = ring_metrics(rings, truth_rings_dev)
        sweep_rows.append(
            {
                "parameter": "decay",
                "value": label,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "detected": len(rings),
            }
        )

    for d in [0.1, 0.3, 0.5]:
        apply_base(engine)
        engine.degree_ratio_threshold = d
        sweep_rows.append(sweep_row("degree_ratio_threshold", d, engine))

    for mval in [1, 2, 3]:
        apply_base(engine)
        engine.min_shared_nodes_to_merge = mval
        sweep_rows.append(sweep_row("min_shared_nodes_to_merge", mval, engine))

    if truncated:
        for mc in [100_000, 200_000, 500_000]:
            apply_base(engine)
            engine.max_candidates = mc
            sweep_rows.append(sweep_row("max_candidates", mc, engine))

    sweep_df = pd.DataFrame(sweep_rows)
    print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = sweep_df.loc[sweep_df["f1"].idxmax()]
    print(f"\nSelected from sweep (max F1): {best['parameter']}={best['value']} "
          f"(F1={best['f1']:.4f})")
    # Apply winners back to base where applicable
    if best["parameter"] == "max_window_hours":
        base["max_window_hours"] = float(best["value"])
    elif best["parameter"] == "degree_ratio_threshold":
        base["degree_ratio_threshold"] = float(best["value"])
    elif best["parameter"] == "min_shared_nodes_to_merge":
        base["min_shared_nodes_to_merge"] = int(best["value"])
    apply_base(engine)

    # --- Step 4 ---
    print("\n" + "=" * 72)
    print("STEP 4 - Ablation study (dev, bootstrap n=500, resample detected)")
    print("=" * 72)

    apply_base(engine)
    full_rings = pipeline_rings(engine, rel_tolerance=base["rel_tolerance"], abs_slack=base["abs_slack"])
    engine.degree_ratio_threshold = 1.01
    no_deg_rings = pipeline_rings(engine, rel_tolerance=base["rel_tolerance"], abs_slack=base["abs_slack"])
    engine.degree_ratio_threshold = base["degree_ratio_threshold"]
    no_dedup_rings = pipeline_rings(
        engine, rel_tolerance=base["rel_tolerance"], abs_slack=base["abs_slack"], dedupe=False
    )
    no_decay_rings = pipeline_rings(
        engine, apply_decay=False, rel_tolerance=base["rel_tolerance"], abs_slack=base["abs_slack"]
    )

    ablations = {
        "Full pipeline": full_rings,
        "No degree-ratio prune": no_deg_rings,
        "No dedup": no_dedup_rings,
        "No decay check": no_decay_rings,
    }

    ablation_table: list[dict] = []
    for name, det in ablations.items():
        m = ring_metrics(det, truth_rings_dev)
        ci = ring_level_bootstrap_ci(det, truth_rings_dev)
        smin, smed, smax = ring_sizes(det)
        tmin, tmed, tmax = ring_sizes(truth_rings_dev)
        rec_pt, rec_lo, rec_hi = ci["recall"]
        if rec_hi > rec_pt + 1e-9:
            print(f"WARNING: recall CI upper ({rec_hi}) > point ({rec_pt}) for {name} — bootstrap bug?")
        ablation_table.append(
            {
                "config": name,
                "precision": fmt_ci(ci["precision"]),
                "recall": fmt_ci(ci["recall"]),
                "f1": fmt_ci(ci["f1"]),
                "account_recall": fmt_ci(ci["account_recall"]),
                "det_min/med/max": f"{smin:.0f}/{smed:.0f}/{smax:.0f}",
                "truth_min/med/max": f"{tmin:.0f}/{tmed:.0f}/{tmax:.0f}",
                "n_det": len(det),
            }
        )
    print(pd.DataFrame(ablation_table).to_string(index=False))

    heavy_truth = fanin_fanout_heavy_rings(engine.graph, truth_rings_dev)
    print(f"\nFan-in/fan-out-heavy dev truth rings (n={len(heavy_truth)}):")
    full_m = ring_metrics(full_rings, truth_rings_dev)
    no_deg_m = ring_metrics(no_deg_rings, truth_rings_dev)
    heavy_full = ring_metrics(
        [r for r in full_rings if frozenset(r) in heavy_truth or any(frozenset(r) >= h for h in heavy_truth)],
        [list(h) for h in heavy_truth],
    ) if heavy_truth else {"recall": float("nan"), "f1": float("nan")}
    # simpler: filter truth to heavy, check recall on those account sets
    heavy_truth_lists = [list(h) for h in heavy_truth]
    hf = ring_metrics(full_rings, heavy_truth_lists)
    hnd = ring_metrics(no_deg_rings, heavy_truth_lists)
    hf_ci = ring_level_bootstrap_ci(full_rings, heavy_truth_lists)
    hnd_ci = ring_level_bootstrap_ci(no_deg_rings, heavy_truth_lists)
    print(f"  Full pipeline recall on heavy subset: {fmt_ci(hf_ci['recall'])}")
    print(f"  No degree prune recall on heavy subset: {fmt_ci(hnd_ci['recall'])}")

    use_no_deg = False
    if heavy_truth and hnd["recall"] > hf["recall"] and hnd_ci["recall"][2] > hf_ci["recall"][1]:
        use_no_deg = True
    degree_prune_decision = (
        "Step 6 will use NO degree-ratio prune (threshold=1.01)."
        if use_no_deg
        else "Step 6 will keep default degree-ratio prune."
    )
    if not heavy_truth:
        degree_prune_decision = "Step 6 keeps default degree-ratio prune (no heavy subset identified)."
    elif not use_no_deg:
        degree_prune_decision += " No-degree-prune recall gain on heavy subset does not clear CI."
    print(f"\nDegree-prune decision: {degree_prune_decision}")

    final_degree = 1.01 if use_no_deg else base["degree_ratio_threshold"]

    # --- Step 5 ---
    print("\n" + "=" * 72)
    print("STEP 5 - Null baseline (dev)")
    print("=" * 72)
    apply_base(engine)
    engine.degree_ratio_threshold = final_degree
    dev_detected = pipeline_rings(engine, rel_tolerance=base["rel_tolerance"], abs_slack=base["abs_slack"])
    real_m = ring_metrics(dev_detected, truth_rings_dev)
    null_rings = sample_topology_matched_null_rings(engine.graph, truth_rings_dev, seed=42)
    null_m = ring_metrics(dev_detected, null_rings)
    sizes_needed = sorted({len(set(r)) for r in truth_rings_dev})
    sizes_null = sorted({len(set(r)) for r in null_rings})
    missing_sizes = [s for s in sizes_needed if s not in sizes_null]
    print(f"Real detector vs dev truth: P={real_m['precision']:.4f} R={real_m['recall']:.4f} F1={real_m['f1']:.4f}")
    print(f"Same detector vs null baseline: P={null_m['precision']:.4f} R={null_m['recall']:.4f} F1={null_m['f1']:.4f}")
    print(f"Null rings sampled: {len(null_rings)} / {len(truth_rings_dev)} truth sizes matched")
    if missing_sizes:
        print(f"GAP: no size-matched topological null for sizes {missing_sizes} "
              f"({len(truth_rings_dev) - len(null_rings)} truth rings skipped)")

    step5_df = pd.DataFrame(
        [
            {"baseline": "dev truth", **{k: real_m[k] for k in ("precision", "recall", "f1", "account_recall")}},
            {"baseline": "topology null", **{k: null_m[k] for k in ("precision", "recall", "f1", "account_recall")}},
        ]
    )
    print(step5_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Step 6 ---
    print("\n" + "=" * 72)
    print("STEP 6 - Held-out test split (NOT re-tuned after seeing this)")
    print("=" * 72)
    if meta["n_test_rings"] == 0:
        print(
            "BLOCKER for ring-level test metrics: 0 CYCLE truth rings have min_ts >= cutoff "
            f"({CUTOFF}). All {meta['n_dev_rings']} labeled CYCLE activity ends before the split. "
            "The prior '10 test rings' were leakage from account co-occurrence, not temporal holdout."
        )
        print("Proceeding with detector run + performance profile only; ring P/R/F1 vs truth N/A.")

    tracemalloc.start()
    t2 = time.perf_counter()
    engine_test = FinancialGraphEngine(
        max_window_hours=base["max_window_hours"],
        degree_ratio_threshold=final_degree,
        min_shared_nodes_to_merge=int(base["min_shared_nodes_to_merge"]),
    )
    engine_test.build_graph_from_transactions(test_df)
    test_detected = pipeline_rings(
        engine_test, rel_tolerance=base["rel_tolerance"], abs_slack=base["abs_slack"]
    )
    t3 = time.perf_counter()
    _, peak_test = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    perf = performance_profile_row(engine_test, "HI-Small-test", t3 - t2, peak_memory_bytes=peak_test)
    print("Performance profile:")
    for k, v in perf.items():
        print(f"  {k}: {v}")

    if truth_rings_test:
        test_ci = ring_level_bootstrap_ci(test_detected, truth_rings_test)
        test_null = sample_topology_matched_null_rings(engine_test.graph, truth_rings_test, seed=42)
        test_null_m = ring_metrics(test_detected, test_null)
        tm = ring_metrics(test_detected, truth_rings_test)
        step6_df = pd.DataFrame(
            [
                {
                    "eval_set": "test truth",
                    "precision": fmt_ci(test_ci["precision"]),
                    "recall": fmt_ci(test_ci["recall"]),
                    "f1": fmt_ci(test_ci["f1"]),
                    "account_recall": fmt_ci(test_ci["account_recall"]),
                },
                {
                    "eval_set": "test null",
                    "precision": f"{test_null_m['precision']:.4f}",
                    "recall": f"{test_null_m['recall']:.4f}",
                    "f1": f"{test_null_m['f1']:.4f}",
                    "account_recall": f"{test_null_m['account_recall']:.4f}",
                },
            ]
        )
        print(step6_df.to_string(index=False))
    else:
        print("Ring-level / account-level test metrics: N/A (0 held-out truth rings).")

    # --- Step 7 ---
    print("\n" + "=" * 72)
    print("STEP 7 - Mode A check")
    print("=" * 72)
    print("SKIPPED: HI-Small is transaction-only; no ticker / Mode A data.")

    print("\n" + "=" * 72)
    print("What would change the conclusion")
    print("=" * 72)
    print("- CIs that span ~0.5 on F1/recall would flip conclusions; inspect ablation table above.")
    print("- With 0 test truth rings, held-out generalization cannot be assessed on this split/cutoff.")
    print("- Null baseline gaps at unmatched ring sizes weaken lift claims for those sizes.")


if __name__ == "__main__":
    main()
