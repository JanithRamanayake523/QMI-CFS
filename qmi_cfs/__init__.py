"""QMI-CFS: Quantum Mutual Information Causal Feature Selection.

A framework for treatment effect estimation in small-sample observational data
using Quantum Mutual Information for confounder identification.

Modules
-------
utils
    Constants, seed management, I/O helpers, and logging utilities.
data_generation
    Synthetic causal dataset generators with known ground-truth confounders.
preprocessing
    Data preprocessing pipeline (imputation, winsorization, normalization,
    outcome standardization).
evaluation
    Evaluation metrics for feature selection quality and treatment effect
    estimation accuracy.

Examples
--------
>>> from qmi_cfs import set_seed, generate_lld, QMIPreprocessor, compute_all_fs_metrics
>>> rng = set_seed(42)
>>> X, T, Y, true_c, true_ate = generate_lld(n_samples=100)
>>> pp = QMIPreprocessor()
>>> data = pp.fit_transform(X, T, Y)
>>> metrics = compute_all_fs_metrics([0, 1], true_c, X.shape[1])
"""

__version__ = '0.1.0'

from qmi_cfs.utils import set_seed, ensure_dir, get_logger, DEFAULT_SEED, QMI_WEIGHTS
from qmi_cfs.data_generation import (
    generate_lld, generate_nlc, generate_hds, generate_cc, generate_wc,
    generate_fixed_ss, generate_ihdp, get_all_datasets,
)
from qmi_cfs.preprocessing import QMIPreprocessor
from qmi_cfs.evaluation import (
    compute_precision, compute_recall, compute_f1, compute_specificity,
    compute_ate_error, compute_all_fs_metrics,
)

__all__ = [
    # Version
    '__version__',
    # Utils
    'set_seed', 'ensure_dir', 'get_logger', 'DEFAULT_SEED', 'QMI_WEIGHTS',
    # Data generation
    'generate_lld', 'generate_nlc', 'generate_hds', 'generate_cc', 'generate_wc',
    'generate_fixed_ss', 'generate_ihdp', 'get_all_datasets',
    # Preprocessing
    'QMIPreprocessor',
    # Evaluation
    'compute_precision', 'compute_recall', 'compute_f1', 'compute_specificity',
    'compute_ate_error', 'compute_all_fs_metrics',
]
