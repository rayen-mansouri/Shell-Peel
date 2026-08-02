import pandas as pd
from datetime import datetime, timedelta
from financial_graph_engine.engine import FinancialGraphEngine

def test_mode_b_pipeline():
    t = datetime(2024, 1, 1, 12, 0, 0)
    # Plant a known 3-node temporal ring: A->B->C->A
    ring_edges = [
        ("A", "B", 100.0, t),
        ("B", "C", 95.0, t + timedelta(hours=1)),
        ("C", "A", 90.0, t + timedelta(hours=2)),
    ]
    
    # Add noise edges that don't form a cycle
    noise_edges = [
        ("X", "Y", 50.0, t),
        ("Y", "Z", 45.0, t + timedelta(hours=1)),
        ("D", "E", 10.0, t),
        ("A", "D", 20.0, t),
    ]
    
    all_edges = ring_edges + noise_edges
    
    df = pd.DataFrame(all_edges, columns=["sender_id", "receiver_id", "amount", "timestamp"])
    
    engine = FinancialGraphEngine()
    engine.build_graph_from_transactions(df)
    rings = engine.detect_circular_routing()
    
    assert len(rings) == 1
    recovered_nodes = set(rings[0])
    assert recovered_nodes == {"A", "B", "C"}
    assert "D" not in recovered_nodes
