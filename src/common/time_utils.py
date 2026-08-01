from __future__ import annotations

import pandas as pd


def coerce_timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")
