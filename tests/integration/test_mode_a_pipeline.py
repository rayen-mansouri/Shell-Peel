import numpy as np
import pandas as pd
from market_anomaly_engine.engine import MarketAnomalyEngine

def test_mode_a_pipeline():
    # Build a 60-row quiet OHLCV frame
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    
    # Add some noise to ensure liquidity conditions are met (mad > min_liquidity=1.0)
    base_close = np.linspace(100, 150, 60)
    noise_close = np.random.normal(0, 5, 60)
    
    # Add noise to volume to ensure mad > min_liquidity=1.0 (on log scale, min_liquidity is compared against log_vol mad)
    # Wait, for log_vol, a mad of 1.0 means volume varies by factor of e~2.7. 
    # Actually, we can just instantiate MarketAnomalyEngine with min_liquidity=0.0 to avoid this complexity for the synthetic test.
    
    df = pd.DataFrame({
        "Date": dates,
        "Adj_Close": np.linspace(100, 105, 60), 
        "Volume": [1000] * 60
    })
    
    # Append 1 row with a 10x volume spike + large return jump
    spike_date = dates[-1] + pd.Timedelta(days=1)
    spike_row = pd.DataFrame({
        "Date": [spike_date],
        "Adj_Close": [150.0],  # big jump
        "Volume": [10000]  # 10x volume
    })
    df = pd.concat([df, spike_row], ignore_index=True)
    
    engine = MarketAnomalyEngine(window=20, z_threshold=3.0, min_liquidity=0.0)
    result = engine.analyze_ticker_data(df)
    
    spike_result = result.iloc[-1]
    quiet_results = result.iloc[:-1]
    
    assert bool(spike_result["Is_Volume_Anomaly"]) is True
    assert bool(spike_result["Is_Volatility_Anomaly"]) is True
    
    assert not quiet_results["Is_Volume_Anomaly"].any()
    assert not quiet_results["Is_Volatility_Anomaly"].any()
