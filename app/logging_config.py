"""
Logging configuration for dbt Co-Work.

This module provides structured logging setup using Python's logging module
with support for both console and file output.

Usage:
    from app.logging_config import setup_logging, get_logger
    
    # Setup once at application start
    setup_logging()
    
    # Get logger in modules
    logger = get_logger(__name__)
    logger.info("Investigation started", extra={"test_id": test_id})
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from app.config import get_settings


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """
    Configure application logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Auto-detected from settings if None.
        log_file: Optional file path for log output
    """
    settings = get_settings()
    
    # Determine log level
    if level is None:
        level = "DEBUG" if settings.debug_mode else "INFO"
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    root_logger.info(f"Logging configured at {level} level")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Configured Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding context to log messages.
    
    Example:
        with LogContext(test_id=test_id, model=model_name):
            logger.info("Starting investigation")
    """
    
    def __init__(self, **context) -> None:
        self.context = context
        self._old_factory = None
    
    def __enter__(self) -> "LogContext":
        self._old_factory = logging.getLogRecordFactory()
        
        context = self.context
        old_factory = self._old_factory
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, *args) -> None:
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)
