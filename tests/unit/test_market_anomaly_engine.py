from __future__ import annotations

import numpy as np
import pandas as pd

from market_anomaly_engine import MarketAnomalyEngine


def _make_price_frame(n_rows: int = 200, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_rows)
    returns = rng.normal(0.0005, 0.012, size=n_rows)
    volume = rng.lognormal(mean=np.log(150_000), sigma=0.2, size=n_rows)

    if n_rows > 60:
        returns[60] += 0.08
        volume[60] *= 4.0
    if n_rows > 150:
        returns[150] += 0.25
        volume[150] *= 8.0

    prices = [25.0]
    for ret in returns[1:]:
        prices.append(prices[-1] * float(np.exp(ret)))

    return pd.DataFrame(
        {
            "Date": dates,
            "Adj_Close": prices,
            "Volume": volume,
        }
    )


def test_percentile_rank_is_expanding_and_not_lookahead_sensitive():
    engine = MarketAnomalyEngine(window=20)
    full_frame = _make_price_frame(200)
    prefix_frame = full_frame.iloc[:65].copy()

    full_scored = engine.analyze_ticker_data(full_frame)
    prefix_scored = engine.analyze_ticker_data(prefix_frame)

    day_60 = pd.Timestamp("2024-03-26")
    full_row = full_scored.loc[full_scored["Date"] == day_60].iloc[0]
    prefix_row = prefix_scored.loc[prefix_scored["Date"] == day_60].iloc[0]

    assert np.isclose(full_row["Vol_Pct_Rank"], prefix_row["Vol_Pct_Rank"], equal_nan=True)
    assert np.isclose(full_row["Volat_Pct_Rank"], prefix_row["Volat_Pct_Rank"], equal_nan=True)


def test_short_series_keeps_warmup_nans_without_errors():
    engine = MarketAnomalyEngine(window=20)
    frame = _make_price_frame(36)
    scored = engine.analyze_ticker_data(frame)

    assert scored["Vol_ZScore_Robust"].isna().all()
    assert scored["Volat_ZScore_Robust"].isna().all()
    assert scored["Vol_Pct_Rank"].isna().all()
    assert scored["Volat_Pct_Rank"].isna().all()
