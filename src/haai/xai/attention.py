from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def pool_over_fields(
    attention: np.ndarray | None,
    token_offsets: Sequence[Tuple[int, int]],
    field_spans: Dict[str, Tuple[int, int]],
    fields: Sequence[str],
) -> np.ndarray:
    weights = np.zeros(len(fields), dtype=float)
    if attention is None or not len(attention):
        return weights
    for index, name in enumerate(fields):
        span = field_spans.get(name)
        if span is None:
            continue
        start, end = span
        mass = [
            float(attention[token])
            for token, (token_start, token_end) in enumerate(token_offsets)
            if token < len(attention) and token_start >= start and token_end <= end + 1
        ]
        weights[index] = float(np.sum(mass)) if mass else 0.0
    total = weights.sum()
    return weights / total if total > 0 else weights


def top_fields(weights: np.ndarray, fields: Sequence[str], k: int = 3) -> Tuple[str, ...]:
    order = np.argsort(-np.asarray(weights, dtype=float))[:k]
    return tuple(str(fields[int(position)]) for position in order)
