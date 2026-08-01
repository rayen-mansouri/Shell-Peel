from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_tabular(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported tabular format: {source.suffix}")


def validate_required_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def discover_amlsim_scenarios(root: str | Path) -> list[Path]:
    base = Path(root)
    scenarios: list[Path] = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and any(grandchild.suffix.lower() == ".csv" for grandchild in child.iterdir() if grandchild.is_file()):
            scenarios.append(child)
    return scenarios


def validate_scenario_separability(root: str | Path) -> list[Path]:
    scenarios = discover_amlsim_scenarios(root)
    if not scenarios:
        raise ValueError(f"No separable AMLSim scenarios found under {root}")
    return scenarios


def load_crosswalk(path: str | Path) -> pd.DataFrame:
    frame = read_tabular(path)
    validate_required_columns(
        frame,
        ["crosswalk_record_id", "entity_a", "entity_b", "link_type", "evidence", "reviewer", "rationale"],
    )
    return frame