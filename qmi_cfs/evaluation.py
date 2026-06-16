"""Evaluation metrics for QMI-CFS experiments.

Provides metrics for:
- Feature selection quality (precision, recall, F1, specificity)
- Treatment effect estimation accuracy (ATE error, PEHE)
"""
import numpy as np
from typing import Dict, List, Optional, Set


def compute_precision(selected_features: List[int], true_confounders: List[int]) -> float:
    """Compute precision: TP / (TP + FP).

    Fraction of selected features that are true confounders.

    Parameters
    ----------
    selected_features : list of int
        Indices of features selected by the method.
    true_confounders : list of int
        Indices of true confounders (ground truth).

    Returns
    -------
    float
        Precision in [0, 1]. Returns 0.0 if no features were selected.
    """
    if len(selected_features) == 0:
        return 0.0
    tp = len(set(selected_features) & set(true_confounders))
    return tp / len(selected_features)


def compute_recall(selected_features: List[int], true_confounders: List[int]) -> float:
    """Compute recall: TP / (TP + FN).

    Fraction of true confounders that were selected.

    Parameters
    ----------
    selected_features : list of int
        Indices of features selected by the method.
    true_confounders : list of int
        Indices of true confounders (ground truth).

    Returns
    -------
    float
        Recall in [0, 1]. Returns 1.0 if there are no true confounders.
    """
    if len(true_confounders) == 0:
        return 1.0
    tp = len(set(selected_features) & set(true_confounders))
    return tp / len(true_confounders)


def compute_f1(selected_features: List[int], true_confounders: List[int]) -> float:
    """Compute F1 score: 2 * precision * recall / (precision + recall).

    Parameters
    ----------
    selected_features : list of int
        Indices of features selected by the method.
    true_confounders : list of int
        Indices of true confounders (ground truth).

    Returns
    -------
    float
        F1 score in [0, 1]. Returns 0.0 if both precision and recall are zero.
    """
    p = compute_precision(selected_features, true_confounders)
    r = compute_recall(selected_features, true_confounders)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_specificity(
    selected_features: List[int],
    true_confounders: List[int],
    n_features: int,
) -> float:
    """Compute specificity: TN / (TN + FP).

    Fraction of non-confounders correctly excluded from selection.

    Parameters
    ----------
    selected_features : list of int
        Indices of features selected by the method.
    true_confounders : list of int
        Indices of true confounders (ground truth).
    n_features : int
        Total number of features.

    Returns
    -------
    float
        Specificity in [0, 1].
    """
    non_confounders = set(range(n_features)) - set(true_confounders)
    selected_set = set(selected_features)
    tn = len(non_confounders - selected_set)
    fp = len(selected_set - set(true_confounders))
    denom = tn + fp
    return tn / denom if denom > 0 else 1.0


def compute_ate_error(estimated_ate: float, true_ate: float) -> float:
    """Compute absolute ATE estimation error.

    Parameters
    ----------
    estimated_ate : float
        Estimated average treatment effect.
    true_ate : float
        True average treatment effect (ground truth).

    Returns
    -------
    float
        Absolute difference |estimated_ate - true_ate|.
    """
    return abs(estimated_ate - true_ate)


def compute_all_fs_metrics(
    selected_features: List[int],
    true_confounders: List[int],
    n_features: int,
) -> Dict[str, float]:
    """Compute all feature selection metrics at once.

    Parameters
    ----------
    selected_features : list of int
        Indices of features selected by the method.
    true_confounders : list of int
        Indices of true confounders (ground truth).
    n_features : int
        Total number of features.

    Returns
    -------
    dict
        Dictionary with keys: 'precision', 'recall', 'f1', 'specificity',
        'n_selected', 'n_true'.
    """
    return {
        'precision': compute_precision(selected_features, true_confounders),
        'recall': compute_recall(selected_features, true_confounders),
        'f1': compute_f1(selected_features, true_confounders),
        'specificity': compute_specificity(selected_features, true_confounders, n_features),
        'n_selected': len(selected_features),
        'n_true': len(true_confounders),
    }


def compute_confusion_matrix(
    selected_features: List[int],
    true_confounders: List[int],
    n_features: int,
) -> Dict[str, int]:
    """Compute confusion matrix elements for feature selection.

    Parameters
    ----------
    selected_features : list of int
        Indices of features selected by the method.
    true_confounders : list of int
        Indices of true confounders (ground truth).
    n_features : int
        Total number of features.

    Returns
    -------
    dict
        Dictionary with keys: 'TP', 'FP', 'TN', 'FN'.
    """
    selected_set = set(selected_features)
    true_set = set(true_confounders)
    all_features = set(range(n_features))

    tp = len(selected_set & true_set)
    fp = len(selected_set - true_set)
    fn = len(true_set - selected_set)
    tn = len(all_features - selected_set - true_set)

    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
