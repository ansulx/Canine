"""Evaluation metrics: AUC, Brier, calibration."""

from typing import List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC AUC. y_pred: probabilities or logits (will use sigmoid if not in [0,1])."""
    if y_pred.min() < 0 or y_pred.max() > 1:
        y_pred = 1.0 / (1.0 + np.exp(-np.asarray(y_pred)))
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_pred))


def auc_pr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average precision (PR AUC)."""
    if y_pred.min() < 0 or y_pred.max() > 1:
        y_pred = 1.0 / (1.0 + np.exp(-np.asarray(y_pred)))
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_pred))


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score (mean squared error of probabilities)."""
    if y_pred.min() < 0 or y_pred.max() > 1:
        y_pred = 1.0 / (1.0 + np.exp(-np.asarray(y_pred)))
    return float(np.mean((y_pred - y_true) ** 2))
