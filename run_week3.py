import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime, timedelta
import tracemalloc
import time
import sys

# Ensure modules can be imported
sys.path.insert(0, str(Path("src").resolve()))
from financial_graph_engine.engine import FinancialGraphEngine
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
    
    # Clean up rings
    cleaned_rings = []
    for r in truth_rings:
        cleaned_rings.append(list(r))
    return cleaned_rings

def run():
    print("Step 0c: Confirming no natural scenario splits...")
    try:
        validate_scenario_separability("C:/Users/Mega Pc/Desktop/shell")
        print("Wait, found scenarios?")
    except ValueError as e:
        print(f"Confirmed no natural scenarios: {e}")

    # Parse Patterns
    patterns_file = "C:/Users/Mega Pc/Desktop/shell/HI-Small_Patterns.txt"
    truth_rings = parse_patterns(patterns_file)
    print(f"Parsed {len(truth_rings)} CYCLE truth rings.")

    # Process Trans in chunks
    trans_file = "C:/Users/Mega Pc/Desktop/shell/HI-Small_Trans.csv"
    dev_chunks = []
    test_chunks = []
    
    # Time window split
    # Day 1-7: Dev (2022/09/01 to 2022/09/07 23:59:59)
    # Day 8-10: Test (2022/09/08 to 2022/09/10 23:59:59)
    cutoff = pd.to_datetime("2022/09/08 00:00:00")
    
    print("Reading transactions...")
    for chunk in pd.read_csv(trans_file, chunksize=500_000):
        # rename columns
        cols = list(chunk.columns)
        # Find first Account and second Account
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
    print(f"Dev rows: {len(dev_df)}, Test rows: {len(test_df)}")

    # Step 1: Build graph on dev
    print("Step 1: Building dev graph...")
    engine_dev = FinancialGraphEngine()
    engine_dev.build_graph_from_transactions(dev_df)
    print(f"Dev graph edges: {engine_dev.graph.number_of_edges()}, nodes: {engine_dev.graph.number_of_nodes()}")
    
    # Step 2: Calibrate
    print("Step 2: Calibrate decay parameters...")
    truth_nodes = set()
    for r in truth_rings:
        truth_nodes.update(r)
    
    truth_amounts = []
    for u, v, data in engine_dev.graph.edges(data=True):
        if u in truth_nodes and v in truth_nodes:
            truth_amounts.append(data["amount"])
    
    if truth_amounts:
        print(f"Truth amounts stats: min={np.min(truth_amounts)}, median={np.median(truth_amounts)}, max={np.max(truth_amounts)}")
    else:
        print("No truth amounts found in dev split!")
        
if __name__ == "__main__":
    run()
