import pandas as pd
import networkx as nx
from pathlib import Path
import sys
import json
import os

sys.path.insert(0, str(Path("src").resolve()))
from financial_graph_engine.engine import FinancialGraphEngine, is_decaying_ring

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
    return [list(r) for r in truth_rings]

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
    
    base_window = 72.0
    base_merge = 2
    base_rel = 0.05
    base_abs = 50.0

    print("Building Dev Graph...")
    engine_dev = FinancialGraphEngine(max_window_hours=base_window, degree_ratio_threshold=1.01, min_shared_nodes_to_merge=base_merge)
    engine_dev.build_graph_from_transactions(dev_df)
    dev_rings = [r for r in engine_dev.detect_temporal_rings() if is_decaying_ring(r, rel_tolerance=base_rel, abs_slack=base_abs)]
    dev_deduped = engine_dev._dedupe_into_rings(dev_rings)

    print("Building Test Graph...")
    engine_test = FinancialGraphEngine(max_window_hours=base_window, degree_ratio_threshold=1.01, min_shared_nodes_to_merge=base_merge)
    engine_test.build_graph_from_transactions(test_df)
    test_rings = [r for r in engine_test.detect_temporal_rings() if is_decaying_ring(r, rel_tolerance=base_rel, abs_slack=base_abs)]
    test_deduped = engine_test._dedupe_into_rings(test_rings)

    out_dir = r"C:\Users\Mega Pc\.gemini\antigravity\brain\f2772c4b-b73a-4ac5-a9fd-855f702deba0\scratch"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "dev_detected_rings.json"), "w") as f:
        json.dump([list(r) for r in dev_deduped], f, indent=2)
    with open(os.path.join(out_dir, "dev_truth_rings.json"), "w") as f:
        json.dump([list(r) for r in truth_rings_dev], f, indent=2)
    with open(os.path.join(out_dir, "test_detected_rings.json"), "w") as f:
        json.dump([list(r) for r in test_deduped], f, indent=2)
    with open(os.path.join(out_dir, "test_truth_rings.json"), "w") as f:
        json.dump([list(r) for r in truth_rings_test], f, indent=2)
        
    print(f"Dumped rings to {out_dir}")

if __name__ == "__main__":
    run()
