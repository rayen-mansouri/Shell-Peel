import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime, timedelta
import tracemalloc
import time
import sys
import json

# Ensure modules can be imported
sys.path.insert(0, str(Path("src").resolve()))
from financial_graph_engine.engine import FinancialGraphEngine, is_decaying_ring
from common.benchmarking import ring_metrics, bootstrap_ci, sample_topology_matched_null_rings, performance_profile_row
from common.ingestion import validate_scenario_separability

def parse_patterns(file_path):
    truth_rings = []
    with open(file_path, "r") as f:
        in_cycle = False
        current_ring = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT - CYCLE"):
                in_cycle = True
                current_ring = []
            elif line.startswith("END LAUNDERING ATTEMPT - CYCLE"):
                in_cycle = False
                if current_ring:
                    truth_rings.append(frozenset(current_ring))
            elif in_cycle:
                parts = line.split(",")
                if len(parts) >= 5:
                    sender = parts[2]
                    receiver = parts[4]
                    current_ring.append(sender)
                    current_ring.append(receiver)
    
    cleaned_rings = []
    for r in truth_rings:
        cleaned_rings.append(list(r))
    return cleaned_rings

def format_ci(est, lo, hi):
    return f"{est:.4f} ({lo:.4f}, {hi:.4f})"

def safe_bootstrap(samples, metric_fn):
    if len(samples) == 0:
        return 0.0, 0.0, 0.0
    return bootstrap_ci(samples, metric_fn, n_resamples=500, ci=0.95, seed=42)

def run():
    print("Loading data...")
    patterns_file = "C:/Users/Mega Pc/Desktop/shell/HI-Small_Patterns.txt"
    truth_rings = parse_patterns(patterns_file)

    trans_file = "C:/Users/Mega Pc/Desktop/shell/HI-Small_Trans.csv"
    dev_chunks = []
    test_chunks = []
    
    cutoff = pd.to_datetime("2022/09/08 00:00:00")
    for chunk in pd.read_csv(trans_file, chunksize=500_000):
        cols = list(chunk.columns)
        acct_indices = [i for i, c in enumerate(cols) if "Account" in c]
        if len(acct_indices) >= 2:
            cols[acct_indices[0]] = "sender_id"
            cols[acct_indices[1]] = "receiver_id"
        chunk.columns = cols
        
        chunk["timestamp"] = pd.to_datetime(chunk["Timestamp"])
        chunk["amount"] = chunk["Amount Paid"]
        
        needed = chunk[["sender_id", "receiver_id", "amount", "timestamp"]]
        dev_chunks.append(needed[needed["timestamp"] < cutoff])
        test_chunks.append(needed[needed["timestamp"] >= cutoff])

    dev_df = pd.concat(dev_chunks, ignore_index=True)
    test_df = pd.concat(test_chunks, ignore_index=True)
    
    def get_valid_rings(df):
        valid_nodes = set(df["sender_id"]) | set(df["receiver_id"])
        res = []
        for r in truth_rings:
            if all(n in valid_nodes for n in r):
                res.append(r)
        return res
    
    truth_rings_dev = get_valid_rings(dev_df)
    truth_rings_test = get_valid_rings(test_df)
    
    engine_dev = FinancialGraphEngine()
    engine_dev.build_graph_from_transactions(dev_df)
    
    base_window = 72.0
    base_degree = 0.3
    base_merge = 2
    base_rel = 0.05
    base_abs = 50.0

    print("--- SWEEP ---")
    # Base
    base_rings = [r for r in engine_dev.detect_temporal_rings() if is_decaying_ring(r, rel_tolerance=base_rel, abs_slack=base_abs)]
    base_deduped = engine_dev._dedupe_into_rings(base_rings)
    base_m = ring_metrics(base_deduped, truth_rings_dev)
    print(f"Base | F1: {base_m['f1']:.4f} | Rec: {base_m['recall']:.4f} | Prec: {base_m['precision']:.4f} | Det: {len(base_deduped)}")

    # Window
    for w in [24, 72, 168]:
        engine_dev.max_window_hours = w
        rings = [r for r in engine_dev.detect_temporal_rings() if is_decaying_ring(r, rel_tolerance=base_rel, abs_slack=base_abs)]
        ded = engine_dev._dedupe_into_rings(rings)
        m = ring_metrics(ded, truth_rings_dev)
        print(f"Window {w}h | F1: {m['f1']:.4f} | Rec: {m['recall']:.4f} | Prec: {m['precision']:.4f} | Det: {len(ded)}")
    engine_dev.max_window_hours = base_window
    
    # Decay
    for r_tol, a_slack in [(base_rel*0.5, base_abs*0.5), (base_rel, base_abs), (base_rel*1.5, base_abs*1.5)]:
        rings = [r for r in engine_dev.detect_temporal_rings() if is_decaying_ring(r, rel_tolerance=r_tol, abs_slack=a_slack)]
        ded = engine_dev._dedupe_into_rings(rings)
        m = ring_metrics(ded, truth_rings_dev)
        print(f"Decay ({r_tol:.3f}, {a_slack:.1f}) | F1: {m['f1']:.4f} | Rec: {m['recall']:.4f} | Prec: {m['precision']:.4f} | Det: {len(ded)}")
        
    # Degree Ratio
    for d in [0.1, 0.3, 0.5]:
        engine_dev.degree_ratio_threshold = d
        rings = [r for r in engine_dev.detect_temporal_rings() if is_decaying_ring(r, rel_tolerance=base_rel, abs_slack=base_abs)]
        ded = engine_dev._dedupe_into_rings(rings)
        m = ring_metrics(ded, truth_rings_dev)
        print(f"Degree {d} | F1: {m['f1']:.4f} | Rec: {m['recall']:.4f} | Prec: {m['precision']:.4f} | Det: {len(ded)}")
    engine_dev.degree_ratio_threshold = base_degree
    
    # Merge threshold
    for m_thresh in [1, 2, 3]:
        engine_dev.min_shared_nodes_to_merge = m_thresh
        ded = engine_dev._dedupe_into_rings(base_rings)
        m = ring_metrics(ded, truth_rings_dev)
        print(f"Merge {m_thresh} | F1: {m['f1']:.4f} | Rec: {m['recall']:.4f} | Prec: {m['precision']:.4f} | Det: {len(ded)}")
    engine_dev.min_shared_nodes_to_merge = base_merge
    
    print("\n--- ABLATION & CIs ---")
    
    # helper for CI
    def print_ci_stats(name, det_rings):
        m = ring_metrics(det_rings, truth_rings_dev)
        
        # recall CI: resample truth rings
        def rec_fn(resampled_truth):
            return ring_metrics(det_rings, resampled_truth)["recall"]
        rec_ci = safe_bootstrap(truth_rings_dev, rec_fn)
        
        # prec CI: resample detected rings
        def prec_fn(resampled_det):
            return ring_metrics(resampled_det, truth_rings_dev)["precision"]
        prec_ci = safe_bootstrap(det_rings, prec_fn)
        
        # f1 CI: we can roughly bootstrap over truth rings and just use original precision if prec is near 0, or just report 0.
        # But wait, we can just resample (truth_rings) for a rough F1 CI assuming detected rings are fixed, or vice versa.
        # Let's resample truth rings and measure F1.
        def f1_fn(resampled_truth):
            return ring_metrics(det_rings, resampled_truth)["f1"]
        f1_ci = safe_bootstrap(truth_rings_dev, f1_fn)
        
        print(f"{name} -> F1: {format_ci(*f1_ci)} | Rec: {format_ci(*rec_ci)} | Prec: {format_ci(*prec_ci)} | Det: {len(det_rings)}")
        
        # Size distribution
        sizes = [len(set(r)) for r in det_rings]
        if sizes:
            print(f"  Det Sizes: min={np.min(sizes)}, median={np.median(sizes)}, max={np.max(sizes)}")
        else:
            print("  Det Sizes: N/A")

    print_ci_stats("Baseline", base_deduped)

    engine_dev.degree_ratio_threshold = 1.01
    rings_no_deg = [r for r in engine_dev.detect_temporal_rings() if is_decaying_ring(r, rel_tolerance=base_rel, abs_slack=base_abs)]
    deduped_no_deg = engine_dev._dedupe_into_rings(rings_no_deg)
    print_ci_stats("No Degree Prune", deduped_no_deg)
    engine_dev.degree_ratio_threshold = base_degree
    
    raw_rings_nodes = [[u for u,v,t,a in r] for r in base_rings]
    print_ci_stats("No Dedup", raw_rings_nodes)
    
    rings_no_decay = engine_dev.detect_temporal_rings()
    deduped_no_decay = engine_dev._dedupe_into_rings(rings_no_decay)
    print_ci_stats("No Decay Check", deduped_no_decay)
    
    # Truth size distribution
    truth_sizes = [len(set(r)) for r in truth_rings_dev]
    print(f"Truth Sizes (Dev): min={np.min(truth_sizes)}, median={np.median(truth_sizes)}, max={np.max(truth_sizes)}")
    
    print("\n--- TEST SPLIT (No Degree Prune) ---")
    engine_test = FinancialGraphEngine(max_window_hours=base_window, degree_ratio_threshold=1.01, min_shared_nodes_to_merge=base_merge)
    engine_test.build_graph_from_transactions(test_df)
    
    test_temp = [r for r in engine_test.detect_temporal_rings() if is_decaying_ring(r, rel_tolerance=base_rel, abs_slack=base_abs)]
    test_deduped = engine_test._dedupe_into_rings(test_temp)
    
    def test_rec_fn(resampled_truth):
        return ring_metrics(test_deduped, resampled_truth)["recall"]
    def test_prec_fn(resampled_det):
        return ring_metrics(resampled_det, truth_rings_test)["precision"]
    def test_f1_fn(resampled_truth):
        return ring_metrics(test_deduped, resampled_truth)["f1"]
        
    print(f"Test -> F1: {format_ci(*safe_bootstrap(truth_rings_test, test_f1_fn))} | "
          f"Rec: {format_ci(*safe_bootstrap(truth_rings_test, test_rec_fn))} | "
          f"Prec: {format_ci(*safe_bootstrap(test_deduped, test_prec_fn))} | "
          f"Det: {len(test_deduped)}")
    
    test_sizes = [len(set(r)) for r in test_deduped]
    print(f"  Det Sizes: min={np.min(test_sizes)}, median={np.median(test_sizes)}, max={np.max(test_sizes)}")
    truth_test_sizes = [len(set(r)) for r in truth_rings_test]
    print(f"  Truth Sizes (Test): min={np.min(truth_test_sizes)}, median={np.median(truth_test_sizes)}, max={np.max(truth_test_sizes)}")

if __name__ == "__main__":
    run()
