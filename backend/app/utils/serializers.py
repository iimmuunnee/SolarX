"""Utilities for serializing NumPy arrays and pandas DataFrames to JSON."""
import numpy as np
import pandas as pd
from typing import Any, List


def numpy_to_list(obj: Any) -> Any:
    """Convert NumPy arrays and pandas objects to Python lists for JSON serialization.

    Args:
        obj: Object to convert (can be numpy array, pandas Series/DataFrame, or nested structure)

    Returns:
        JSON-serializable Python object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_list(item) for item in obj]
    else:
        return obj


def ensure_finite(values: List[float], default: float = 0.0) -> List[float]:
    """Replace NaN and Inf values with default value.

    Args:
        values: List of float values
        default: Default value to use for NaN/Inf

    Returns:
        List with NaN/Inf replaced
    """
    return [default if not np.isfinite(v) else v for v in values]
