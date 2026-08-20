import pandas as pd
import numpy as np
import datetime
import random
import os
import sys

sys.path.append(os.path.abspath("src"))
from financial_graph_engine.engine import FinancialGraphEngine

def load_and_split():
    print("Loading data...")
    df = pd.read_csv('../HI-Small_Trans.csv')
    df['Timestamp_obj'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
    
    # Parse patterns for CYCLE
    cycle_rings = []
    current_ring = []
    in_cycle = False
    with open('../HI-Small_Patterns.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('BEGIN LAUNDERING ATTEMPT - CYCLE'):
                in_cycle = True
                continue
            if line.startswith('END LAUNDERING ATTEMPT - CYCLE'):
                in_cycle = False
                if current_ring:
                    cycle_rings.append(current_ring)
                current_ring = []
                continue
            if in_cycle and line:
                parts = line.split(',')
                # Format: Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
                # Create a tuple identifier
                ts = parts[0]
                from_bank = parts[1]
                from_acc = parts[2]
                to_bank = parts[3]
                to_acc = parts[4]
                amt_paid = float(parts[7])
                current_ring.append((ts, from_bank, from_acc, to_bank, to_acc, amt_paid))

    print(f"Parsed {len(cycle_rings)} CYCLE rings.")
    
    # Map to DataFrame indices
    # We will iterate through each ring and find its transactions in df
    ring_indices = []
    for r_idx, ring in enumerate(cycle_rings):
        indices_for_this_ring = []
        for hop in ring:
            ts, fb, fa, tb, ta, amt = hop
            # Find in df
            mask = (df['Timestamp'] == ts) & (df['From Bank'] == fb) & (df['Account'] == fa) & \
                   (df['To Bank'] == tb) & (df['Account.1'] == ta) & (np.isclose(df['Amount Paid'], amt))
            matches = df[mask].index.tolist()
            if not matches:
                print(f"WARNING: No match for hop {hop}")
            else:
                indices_for_this_ring.append(matches[0]) # take the first match
        ring_indices.append(indices_for_this_ring)
        
    # Random split of ring IDENTITIES
    random.seed(42)
    dev_ring_ids = set(random.sample(range(len(cycle_rings)), int(0.74 * len(cycle_rings)))) # 14 to dev
    test_ring_ids = set(range(len(cycle_rings))) - dev_ring_ids
    
    dev_trans_indices = set()
    for i in dev_ring_ids:
        dev_trans_indices.update(ring_indices[i])
        
    test_trans_indices = set()
    for i in test_ring_ids:
        test_trans_indices.update(ring_indices[i])
        
    # Check leakage
    leakage = dev_trans_indices.intersection(test_trans_indices)
    assert len(leakage) == 0, "LEAKAGE DETECTED"
    
    print(f"Assigned {len(dev_ring_ids)} rings to dev, {len(test_ring_ids)} to test.")
    
    all_cycle_indices = dev_trans_indices.union(test_trans_indices)
    background_mask = ~df.index.isin(all_cycle_indices)
    
    # Split background by time cutoff (say 2022-09-06 00:00:00)
    cutoff = pd.to_datetime('2022-09-06 00:00:00')
    bg_dev_mask = background_mask & (df['Timestamp_obj'] <= cutoff)
    bg_test_mask = background_mask & (df['Timestamp_obj'] > cutoff)
    
    dev_df = pd.concat([df.loc[list(dev_trans_indices)], df[bg_dev_mask]]).sort_index()
    test_df = pd.concat([df.loc[list(test_trans_indices)], df[bg_test_mask]]).sort_index()
    
    print(f"Dev DF shape: {dev_df.shape}")
    print(f"Test DF shape: {test_df.shape}")
    
    return dev_df, test_df, dev_ring_ids, test_ring_ids, cycle_rings, ring_indices

if __name__ == '__main__':
    load_and_split()
