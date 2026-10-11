"""Evaluation metrics for GIEOK."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Dict, Tuple


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Predicted labels (0=Healthy, 1=AD)
        targets: Ground truth labels

    Returns:
        Dictionary with metrics
    """
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "f1_macro": f1_score(targets, predictions, average='macro'),
        "f1_weighted": f1_score(targets, predictions, average='weighted'),
    }

    # Per-class metrics
    cm = confusion_matrix(targets, predictions)

    # Healthy (class 0)
    if cm.shape[0] > 0:
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]

        metrics["healthy_precision"] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics["healthy_recall"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["healthy_f1"] = 2 * (metrics["healthy_precision"] * metrics["healthy_recall"]) / \
                               (metrics["healthy_precision"] + metrics["healthy_recall"] + 1e-8)

        # AD (class 1)
        metrics["ad_precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics["ad_recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics["ad_f1"] = 2 * (metrics["ad_precision"] * metrics["ad_recall"]) / \
                          (metrics["ad_precision"] + metrics["ad_recall"] + 1e-8)

    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(targets, predictions)


def print_metrics(metrics: Dict[str, float], language: str = "") -> None:
    """Pretty-print metrics."""
    if language:
        print(f"\n{language.upper()}")
        print("-" * 60)

    for metric_name, value in sorted(metrics.items()):
        print(f"  {metric_name:20s}: {value:.4f}")
