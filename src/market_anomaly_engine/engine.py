from __future__ import annotations

import numpy as np
import pandas as pd


class MarketAnomalyEngine:
    def __init__(self, window: int = 20, z_threshold: float = 3.0, min_liquidity: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold
        self.min_liquidity = min_liquidity

    def analyze_ticker_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"Date", "Adj_Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.sort_values("Date").reset_index(drop=True)

        log_vol = np.log1p(out["Volume"])
        vol_med = log_vol.rolling(self.window).median()
        vol_mad = (log_vol - vol_med).abs().rolling(self.window).median()
        out["Vol_ZScore_Robust"] = (log_vol - vol_med) / (1.4826 * vol_mad + 1e-8)
        out["Vol_Pct_Rank"] = out["Vol_ZScore_Robust"].abs().rank(pct=True)

        out["Returns"] = np.log(out["Adj_Close"] / out["Adj_Close"].shift(1))
        ret_med = out["Returns"].rolling(self.window).median()
        ret_mad = (out["Returns"] - ret_med).abs().rolling(self.window).median()
        out["Volat_ZScore_Robust"] = (out["Returns"] - ret_med) / (1.4826 * ret_mad + 1e-8)
        out["Volat_Pct_Rank"] = out["Volat_ZScore_Robust"].abs().rank(pct=True)

        # Gate volume and returns anomalies independently on their own liquidity axis.
        illiquid_volume = vol_mad < self.min_liquidity
        illiquid_returns = ret_mad < self.min_liquidity

        out["Is_Volume_Anomaly"] = (out["Vol_ZScore_Robust"].abs() > self.z_threshold) & ~illiquid_volume
        out["Is_Volatility_Anomaly"] = (out["Volat_ZScore_Robust"].abs() > self.z_threshold) & ~illiquid_returns
        out["Low_Liquidity_Volume_Flag"] = illiquid_volume
        out["Low_Liquidity_Returns_Flag"] = illiquid_returns

        return out
