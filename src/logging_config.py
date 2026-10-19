"""Logging setup: console + rotating file (logs/api.log)."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src import config

FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if getattr(root, "_heart_configured", False):
        return
    root.setLevel(level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(console)
    log_dir = Path(log_dir or config.LOG_DIR)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_dir / "api.log", maxBytes=2_000_000, backupCount=3)
        fh.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(fh)
    except OSError:  # read-only filesystem etc. - console logging is enough
        root.warning("File logging disabled (cannot write to %s)", log_dir)
    root._heart_configured = True  # type: ignore[attr-defined]
