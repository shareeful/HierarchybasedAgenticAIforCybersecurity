from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def unique_records(pairs: pd.DataFrame) -> pd.DataFrame:
    return pairs.drop_duplicates(subset="cve_id").reset_index(drop=True)


def aggregate_to_records(pairs: pd.DataFrame, score_column: str, method: str = "max") -> pd.DataFrame:
    grouped = pairs.groupby("cve_id", observed=True)
    score = getattr(grouped[score_column], method)()
    label = grouped["label"].first()
    return pd.DataFrame(
        {"cve_id": score.index, "score": score.to_numpy(dtype=float), "label": label.to_numpy()}
    )


def matrix_for(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    columns = []
    for name in fields:
        values = frame[name]
        if values.dtype == bool:
            columns.append(values.to_numpy(dtype=float))
        else:
            columns.append(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float))
    return np.column_stack(columns)


def stratified_bootstrap(frame: pd.DataFrame, rng: np.random.Generator, strata: str = "label") -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby(strata, observed=True):
        index = group.index.to_numpy()
        picked = rng.choice(index, size=len(index), replace=True)
        parts.append(frame.loc[picked])
    return pd.concat(parts).reset_index(drop=True)
