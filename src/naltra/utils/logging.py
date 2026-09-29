"""Central logging configuration."""

import logging
import os


def configure_logging() -> None:
    """Configure process logging from NALTRA_LOG_LEVEL."""
    level_name = os.getenv("NALTRA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a conventional named logger."""
    return logging.getLogger(name)
