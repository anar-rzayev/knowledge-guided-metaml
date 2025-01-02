from typing import Callable

import numpy as np
from scipy.stats import spearmanr

def check_shapes(predictions: np.ndarray, targets: np.ndarray) -> None:
    """Ensure predictions and targets have the same shape."""
    assert (
        predictions.shape == targets.shape
    ), f"Predictions shape {predictions.shape} and targets shape {targets.shape} don't match"

def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate mean squared error (MSE)."""
    check_shapes(predictions, targets)
    return ((targets - predictions) ** 2).mean(0)

def standard_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate standardized mean squared error."""
    target_mean = targets.mean(0)
    target_std = targets.std(0) if len(targets) > 1 else 1.0
    standard_predictions = (predictions - target_mean) / target_std
    standard_targets = (targets - target_mean) / target_std
    return mse(standard_predictions, standard_targets)

def norm_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate normalized mean squared error."""
    avg_target_l1 = np.abs(targets).mean(0)
    norm_predictions = predictions / avg_target_l1
    norm_targets = targets / avg_target_l1
    return mse(norm_predictions, norm_targets)

def spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Spearman rank correlation coefficient."""
    check_shapes(predictions, targets)
    return spearmanr(targets, predictions)[0]

def get_metric_function(metric_name: str) -> Callable:
    """Retrieve the metric function by name."""
    if metric_name == "mse":
        return mse
    elif metric_name == "standardized_mse":
        return standard_mse
    elif metric_name == "normalized_mse":
        return norm_mse
    elif metric_name == "spearman":
        return spearman
    else:
        raise ValueError(f"Unknown metric_name {metric_name}")
