"""Model-agnostic metrics for multi-label predictions, shared by the CNN and the classic model.

Every function takes y_true (n, len(SUPERCLASSES)) as 0/1 and y_prob (same shape) as probabilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

from src.preprocessing import SUPERCLASSES

DEFAULT_THRESHOLD = 0.5
THRESHOLD_CANDIDATES = np.round(np.arange(0.05, 0.96, 0.01), 2)


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-class threshold that maximizes F1. Fit on validation data only, never on test."""
    thresholds = np.empty(len(SUPERCLASSES))
    for c in range(len(SUPERCLASSES)):
        scores = [f1_score(y_true[:, c], y_prob[:, c] >= t, zero_division=0)
                  for t in THRESHOLD_CANDIDATES]
        thresholds[c] = THRESHOLD_CANDIDATES[np.argmax(scores)]
    return thresholds


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    thresholds: np.ndarray | float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Per-class AUC, precision, recall and F1, plus a macro-average row."""
    y_pred = y_prob >= thresholds
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics = pd.DataFrame({
        "threshold": np.broadcast_to(thresholds, len(SUPERCLASSES)),
        "auc": roc_auc_score(y_true, y_prob, average=None),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }, index=SUPERCLASSES)
    metrics.loc["macro"] = metrics.drop(columns="support").mean().tolist() + [support.sum()]
    metrics.loc["macro", "threshold"] = np.nan  # an average of thresholds has no meaning
    return metrics
