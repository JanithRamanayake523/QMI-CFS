"""Utilities for QMI-CFS: constants, seed management, I/O helpers."""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

DEFAULT_SEED: int = 42
DELTA: float = 0.05
QMI_WEIGHTS: Dict[str, Tuple[float, float, float]] = {
    'standard': (0.4, 0.4, 0.2),
    'conservative': (1.0 / 3, 1.0 / 3, 1.0 / 3),
    'precision': (0.2, 0.6, 0.2),
}


def set_seed(seed: Optional[int] = None) -> np.random.Generator:
    """Set random seed and return Generator instance.
    
    Parameters
    ----------
    seed : int, optional
        Random seed value. If None, uses DEFAULT_SEED (42).
    
    Returns
    -------
    np.random.Generator
        Seeded random number generator.
    """
    seed = seed if seed is not None else DEFAULT_SEED
    return np.random.default_rng(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist.
    
    Parameters
    ----------
    path : str or Path
        Directory path to create.
    
    Returns
    -------
    Path
        The created/existing directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dict as JSON file.
    
    Parameters
    ----------
    data : dict
        Dictionary to serialize.
    path : str or Path
        Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file into dict.
    
    Parameters
    ----------
    path : str or Path
        Path to JSON file.
    
    Returns
    -------
    dict
        Parsed JSON content.
    """
    with open(path, 'r') as f:
        return json.load(f)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get logger with standard formatting.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__).
    level : int
        Logging level (default: INFO).
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
