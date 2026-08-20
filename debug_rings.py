import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
import pandas as pd

def parse_patterns(file_path):
    truth_rings = []
    with open(file_path, encoding="utf-8") as f:
        in_cycle = False
        current_accounts = []
        current_timestamps = []
        current_hops = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT - CYCLE"):
                in_cycle = True
                current_accounts, current_timestamps, current_hops = [], [], []
            elif line.startswith("END LAUNDERING ATTEMPT - CYCLE"):
                in_cycle = False
                if current_accounts:
                    truth_rings.append({
                        "accounts": frozenset(current_accounts),
                        "min_ts": min(current_timestamps),
                        "max_ts": max(current_timestamps),
                        "hops": current_hops,
                    })
            elif in_cycle:
                parts = line.split(",")
                if len(parts) >= 5:
                    ts = pd.to_datetime(parts[0].strip())
                    sender = parts[2].strip()
                    receiver = parts[4].strip()
                    amount_paid = float(parts[7].strip())
                    current_timestamps.append(ts)
                    current_accounts.append(sender)
                    current_accounts.append(receiver)
                    current_hops.append({
                        "timestamp": parts[0].strip(),
                        "sender": sender,
                        "receiver": receiver,
                        "amount_paid": amount_paid,
                    })
    return truth_rings

rings = parse_patterns(Path("../HI-Small_Patterns.txt"))
rings.sort(key=lambda r: r["min_ts"])

from collections import Counter
acct_counts = Counter()
for i, r in enumerate(rings):
    acct_counts[r["accounts"]] += 1

dups = {k: v for k, v in acct_counts.items() if v > 1}
print("Duplicate account frozensets:", len(dups))
for accs, count in dups.items():
    indices = [i for i, r in enumerate(rings) if r["accounts"] == accs]
    print("  Accounts (first 3):", sorted(accs)[:3])
    print("  Ring indices:", indices, "count:", count)
    for idx in indices:
        r = rings[idx]
        print("    Ring", idx, "min_ts:", r["min_ts"], "max_ts:", r["max_ts"], "n_hops:", len(r["hops"]))

# Also print sizes
print("\nAll ring account-set sizes:")
for i, r in enumerate(rings):
    print(f"  Ring {i}: {len(r['accounts'])} unique accounts (frozenset size)")
