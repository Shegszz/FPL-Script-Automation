from __future__ import annotations

from typing import Iterable

import numpy as np


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean Absolute Error."""
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)

    if len(actual) == 0:
        return float("nan")

    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Root Mean Squared Error."""
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)

    if len(actual) == 0:
        return float("nan")

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mean_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Signed prediction error: prediction - actual."""
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)

    if len(actual) == 0:
        return float("nan")

    return float(np.mean(predicted - actual))


def within_tolerance(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    tolerance: float = 2.0,
) -> float:
    """Fraction of predictions within +/- tolerance points."""
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)

    if len(actual) == 0:
        return float("nan")

    return float(np.mean(np.abs(actual - predicted) <= tolerance))


def evaluate_predictions(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict:
    """Calculate core forecasting metrics."""
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mean_error": round(mean_error(y_true, y_pred), 4),
        "within_2_points": round(
            within_tolerance(y_true, y_pred, tolerance=2.0),
            4,
        ),
    }
