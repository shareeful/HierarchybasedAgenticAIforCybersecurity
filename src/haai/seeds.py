from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np


def _stable_hash(name: str) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


@dataclass(frozen=True)
class SeedBook:
    master_seed: int

    def stream(self, name: str, run_index: int = 0) -> np.random.Generator:
        entropy = [self.master_seed, _stable_hash(name), int(run_index)]
        return np.random.default_rng(np.random.SeedSequence(entropy))

    def integer(self, name: str, run_index: int = 0) -> int:
        return int(self.stream(name, run_index).integers(0, 2**31 - 1))

    def run_seeds(self, name: str, n_runs: int) -> Dict[int, int]:
        return {i: self.integer(name, i) for i in range(n_runs)}


def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
