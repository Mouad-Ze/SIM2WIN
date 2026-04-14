"""
Logging Configuration for Sim2Win
Provides centralized logging setup for all components
"""

import logging
import sys
from pathlib import Path
from config import LOGGING_CONFIG


def setup_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOGGING_CONFIG['level']))
    
    # File handler (optional - creates log file)
    try:
        log_file = Path(LOGGING_CONFIG['log_file'])
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    # Formatter
    formatter = logging.Formatter(LOGGING_CONFIG['format'])
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger


# Create module-level logger
logger = setup_logger(__name__)
