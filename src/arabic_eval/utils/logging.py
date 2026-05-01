"""Structured logging setup."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "arabic_eval",
    log_file: Optional[str | Path] = None,
    error_log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger with console + optional file handlers.

    If ``error_log_file`` is given, a second FileHandler is attached that
    only writes records at WARNING or higher — a quick way to see whether
    anything went wrong during a long run without grepping the full log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if error_log_file is not None:
        Path(error_log_file).parent.mkdir(parents=True, exist_ok=True)
        eh = logging.FileHandler(error_log_file)
        eh.setLevel(logging.WARNING)
        eh.setFormatter(fmt)
        logger.addHandler(eh)

    return logger
