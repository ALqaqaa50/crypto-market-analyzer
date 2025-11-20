"""
Centralized logging configuration
"""
import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: Optional[int] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Get or create a configured logger.
    
    Args:
        name: Logger name (usually module name)
        level: Logging level (defaults to INFO)
        log_format: Custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Set level
    if level is None:
        level = logging.INFO
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    if log_format is None:
        log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
    
    formatter = logging.Formatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def set_global_log_level(level: int):
    """Set log level for all loggers"""
    logging.root.setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)