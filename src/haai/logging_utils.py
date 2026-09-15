from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str = "haai", log_dir: str | Path | None = None) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        logger.setLevel(logging.INFO)
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(stream)
        if log_dir is not None:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path / "haai.log")
            file_handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s %(name)s %(message)s")
            )
            logger.addHandler(file_handler)
        logger.propagate = False
        _CONFIGURED = True
    return logger


@contextmanager
def timed(label: str, logger: logging.Logger | None = None):
    logger = logger or get_logger()
    start = time.perf_counter()
    logger.info("%s ...", label)
    try:
        yield
    finally:
        logger.info("%s done in %.2fs", label, time.perf_counter() - start)
