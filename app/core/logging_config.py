"""Application logging configuration for Crawlbot."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Optional

from .config import Settings, get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_logging_initialized: bool = False


def setup_logging(settings: Optional[Settings] = None) -> Callable[[str], logging.Logger]:
    """Configure structured logging for Crawlbot with console and file handlers.

    Creates the `logs/` directory if needed, sets the logging level from
    application settings, and attaches formatted stream and rotating file handlers
    to the root logger.

    Args:
        settings: Optional Settings instance. If not provided, retrieved via get_settings().

    Returns:
        Callable[[str], logging.Logger]: Logger factory function returning named loggers.
    """
    global _logging_initialized

    if settings is None:
        settings = get_settings()

    log_level_name = getattr(settings, "log_level", "INFO").upper()
    level = getattr(logging, log_level_name, logging.INFO)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / "crawlbot.log"

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to prevent duplicate logs on re-initialization
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating File Handler (10MB max, 5 backups, UTF-8)
    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _logging_initialized = True

    return get_logger


def get_logger(name: str = "crawlbot") -> logging.Logger:
    """Retrieve a configured logger by name.

    If logging has not yet been initialized, setup_logging() is automatically executed.

    Args:
        name: Name for the logger instance (defaults to 'crawlbot').

    Returns:
        logging.Logger: Configured logger instance.
    """
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()
    return logging.getLogger(name)
