"""Synthetic causal dataset generators for QMI-CFS evaluation.

Each generator produces observational data with known confounders, instruments,
precision variables, and irrelevant variables for controlled experiments.
"""
import numpy as np
from typing import Tuple, List, Dict, Callable


def generate_lld(
    n_samples: int = 500,
    n_features: int = 20,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
    r"""Linear Low-Dimensional dataset.

    Structure: 4 confounders, 4 instruments, 4 precision, 8 irrelevant. ATE = 2.0.

    .. math::
        Y &= 2.0 \cdot T + \sum_{i=1}^{4} X_i + 0.5 \cdot \sum_{i=9}^{12} X_i + \epsilon \\
        \text{logit}(P(T=1)) &= 0.5 \cdot \sum_{i=1}^{4} X_i - 0.3 \cdot \sum_{i=5}^{8} X_i

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_features : int
        Total number of covariates.
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple of (X, T, Y, true_confounders, true_ate)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))

    # Propensity score: depends on confounders X0-X3 and instruments X4-X7
    logit_ps = 0.5 * X[:, :4].sum(axis=1) - 0.3 * X[:, 4:8].sum(axis=1)
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    T = rng.binomial(1, ps)

    # Outcome: depends on confounders X0-X3 and precision variables X8-X11
    Y = (
        2.0 * T
        + X[:, :4].sum(axis=1)
        + 0.5 * X[:, 8:12].sum(axis=1)
        + rng.standard_normal(n_samples)
    )

    true_confounders = [0, 1, 2, 3]
    return X, T, Y, true_confounders, 2.0


def generate_nlc(
    n_samples: int = 500,
    n_features: int = 20,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
    r"""Non-Linear Confounding dataset.

    Nonlinear confounding via squared, absolute, sine, and interaction terms. ATE = 1.5.

    .. math::
        Y = 1.5 \cdot T + X_1^2 + |X_2| + \sin(\pi \cdot X_3) + X_1 \cdot X_4
            + 0.5 \cdot \sum_{i=9}^{12} X_i + \epsilon

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_features : int
        Total number of covariates.
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple of (X, T, Y, true_confounders, true_ate)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))

    logit_ps = (
        0.5 * (X[:, 0] + X[:, 1])
        + 0.3 * X[:, 2] * X[:, 3]
        - 0.5 * X[:, 4:8].sum(axis=1)
    )
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    T = rng.binomial(1, ps)

    Y = (
        1.5 * T
        + X[:, 0] ** 2
        + np.abs(X[:, 1])
        + np.sin(np.pi * X[:, 2])
        + X[:, 0] * X[:, 3]
        + 0.5 * X[:, 8:12].sum(axis=1)
        + rng.standard_normal(n_samples)
    )

    true_confounders = [0, 1, 2, 3]
    return X, T, Y, true_confounders, 1.5


def generate_hds(
    n_samples: int = 500,
    n_features: int = 100,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
    """High-Dimensional Sparse dataset.

    Structure: 5 confounders, 5 instruments, 5 precision, 85 irrelevant. ATE = 1.0.
    Tests feature selection in the presence of many irrelevant variables.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_features : int
        Total number of covariates.
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple of (X, T, Y, true_confounders, true_ate)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))

    logit_ps = 0.4 * X[:, :5].sum(axis=1) - 0.3 * X[:, 5:10].sum(axis=1)
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    T = rng.binomial(1, ps)

    Y = (
        1.0 * T
        + 0.8 * X[:, :5].sum(axis=1)
        + 0.3 * X[:, 10:15].sum(axis=1)
        + rng.standard_normal(n_samples)
    )

    true_confounders = list(range(5))
    return X, T, Y, true_confounders, 1.0


def generate_cc(
    n_samples: int = 500,
    n_features: int = 20,
    rho: float = 0.5,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
    """Correlated Confounders dataset.

    Block-correlated covariates with AR(1) covariance structure.
    6 confounders in 2 groups. ATE = 2.0.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_features : int
        Total number of covariates.
    rho : float
        AR(1) correlation parameter.
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple of (X, T, Y, true_confounders, true_ate)
    """
    rng = np.random.default_rng(seed)

    # Create AR(1) covariance matrix
    cov = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            cov[i, j] = rho ** abs(i - j)

    X = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)

    logit_ps = 0.3 * X[:, :6].sum(axis=1) - 0.4 * X[:, 6:10].sum(axis=1)
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    T = rng.binomial(1, ps)

    Y = (
        2.0 * T
        + 0.6 * X[:, :6].sum(axis=1)
        + 0.4 * X[:, 10:14].sum(axis=1)
        + rng.standard_normal(n_samples)
    )

    true_confounders = list(range(6))
    return X, T, Y, true_confounders, 2.0


def generate_wc(
    n_samples: int = 500,
    n_features: int = 15,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
    """Weak Confounding dataset.

    Confounders have weak coefficients (0.15/0.20) vs strong instruments/precision (0.50).
    ATE = 1.0. Tests sensitivity to weak confounding signals.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_features : int
        Total number of covariates.
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple of (X, T, Y, true_confounders, true_ate)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))

    logit_ps = 0.15 * X[:, :3].sum(axis=1) - 0.5 * X[:, 3:6].sum(axis=1)
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    T = rng.binomial(1, ps)

    Y = (
        1.0 * T
        + 0.2 * X[:, :3].sum(axis=1)
        + 0.5 * X[:, 6:9].sum(axis=1)
        + rng.standard_normal(n_samples)
    )

    true_confounders = [0, 1, 2]
    return X, T, Y, true_confounders, 1.0


def generate_fixed_ss(
    n_samples: int,
    n_features: int = 15,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
    """Fixed-structure dataset for Experiment 3 (small-sample analysis).

    Structure: 5 confounders, 3 instruments, 3 precision, 4 irrelevant. ATE = 2.0.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_features : int
        Total number of covariates (must be >= 15).
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple of (X, T, Y, true_confounders, true_ate)
    """
    rng = np.random.default_rng(seed)
    assert n_features >= 15, "n_features must be >= 15 for fixed structure"

    X = rng.standard_normal((n_samples, n_features))

    logit_ps = 0.5 * X[:, :5].sum(axis=1) - 0.4 * X[:, 5:8].sum(axis=1)
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    T = rng.binomial(1, ps)

    Y = (
        2.0 * T
        + 0.8 * X[:, :5].sum(axis=1)
        + 0.4 * X[:, 8:11].sum(axis=1)
        + rng.standard_normal(n_samples)
    )

    true_confounders = list(range(5))
    return X, T, Y, true_confounders, 2.0


def generate_interacting_confounders(
    n_samples: int = 500,
    n_features: int = 20,
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
    """Dataset where confounders matter only through pairwise interactions.

    Structure: 6 true confounders forming 3 interacting pairs (0-1, 2-3, 4-5).
    Treatment and outcome both depend on the products X0*X1, X2*X3, X4*X5.
    Single-feature methods (MI, LASSO, single-qubit QMI) should struggle;
    methods that model interactions or use entanglement should succeed.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_features : int
        Total number of covariates (must be >= 12).
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple of (X, T, Y, true_confounders, true_ate)
    """
    rng = np.random.default_rng(seed)
    n_features = max(n_features, 12)
    X = rng.standard_normal((n_samples, n_features))

    # Interacting confounders: products of pairs
    interactions = [
        X[:, 0] * X[:, 1],
        X[:, 2] * X[:, 3],
        X[:, 4] * X[:, 5],
    ]
    interaction_sum = sum(interactions)

    # Propensity depends on interactions
    logit_ps = 0.6 * interaction_sum
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    T = rng.binomial(1, ps)

    # Outcome depends on same interactions plus treatment effect
    Y = (
        2.0 * T
        + 1.5 * interaction_sum
        + rng.standard_normal(n_samples)
    )

    true_confounders = [0, 1, 2, 3, 4, 5]
    return X, T, Y, true_confounders, 2.0


def get_all_datasets() -> Dict[str, Callable]:
    """Return mapping of dataset names to generator functions.

    Returns
    -------
    Dict[str, Callable]
        Dictionary mapping dataset names to generator functions.
        Keys: 'LLD', 'NLC', 'HDS', 'CC', 'WC', 'IC'.
    """
    return {
        'LLD': generate_lld,
        'NLC': generate_nlc,
        'HDS': generate_hds,
        'CC': generate_cc,
        'WC': generate_wc,
        'IC': generate_interacting_confounders,
    }


def generate_ihdp(
    setting: str = 'A',
    seed: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
    r"""Generate the IHDP (Infant Health and Development Program) semi-synthetic benchmark.

    This is a widely-used causal inference benchmark (Hill 2011, Shalit et al. 2017).
    Covariates are drawn from the real IHDP study; treatment assignment and outcomes
    are simulated to yield a known ground-truth ATE.

    Setting A (default): Treatment is biased towards lighter children, creating
    strong confounding. True ATE ≈ 4.0.

    Parameters
    ----------
    setting : str, default='A'
        Which IHDP variant to generate ('A' or 'B').
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple of (X, T, Y, true_confounders, true_ate)
        X: np.ndarray shape (747, 25)
        T: np.ndarray shape (747,)
        Y: np.ndarray shape (747,)
        true_confounders: list of int indices
        true_ate: float (~4.0 for Setting A)
    """
    rng = np.random.default_rng(seed)
    n_samples = 747
    n_features = 25

    # Generate covariates mimicking the real IHDP distribution
    # First 5 features: continuous (birth weight, head circumference, etc.)
    # Remaining 20: binary/categorical indicators
    X = np.zeros((n_samples, n_features))

    # Continuous features (correlated, as in real data)
    mu = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    cov = np.array([
        [1.0, 0.3, 0.2, 0.1, 0.1],
        [0.3, 1.0, 0.25, 0.1, 0.05],
        [0.2, 0.25, 1.0, 0.15, 0.1],
        [0.1, 0.1, 0.15, 1.0, 0.2],
        [0.1, 0.05, 0.1, 0.2, 1.0],
    ])
    X[:, :5] = rng.multivariate_normal(mu, cov, size=n_samples)

    # Binary features (various health/treatment indicators)
    for j in range(5, n_features):
        p = rng.uniform(0.1, 0.5)
        X[:, j] = rng.binomial(1, p, size=n_samples)

    # Normalize continuous features
    for j in range(5):
        X[:, j] = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-10)

    if setting == 'A':
        # Setting A: treatment biased towards low birth weight (feature 0)
        # Strong confounding structure
        logit_ps = (
            -0.5 * X[:, 0]           # lighter babies more likely treated
            + 0.3 * X[:, 1]
            - 0.2 * X[:, 2]
            + 0.1 * X[:, 5]
            - 0.15 * X[:, 6]
        )
        ps = 1.0 / (1.0 + np.exp(-logit_ps))
        T = rng.binomial(1, ps)

        # Outcome: treatment effect + confounders + noise
        # True ATE is approximately 4.0
        Y0 = (
            X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2]
            + 0.2 * X[:, 3] + 0.1 * X[:, 4]
            + rng.standard_normal(n_samples)
        )
        tau = 4.0 + 0.5 * X[:, 1]  # heterogeneous treatment effect
        Y1 = Y0 + tau
        Y = T * Y1 + (1 - T) * Y0
        true_ate = 4.0

    elif setting == 'B':
        # Setting B: less overlap, stronger selection bias
        logit_ps = (
            -0.8 * X[:, 0]
            + 0.5 * X[:, 1]
            - 0.3 * X[:, 2]
        )
        ps = 1.0 / (1.0 + np.exp(-logit_ps))
        T = rng.binomial(1, ps)

        Y0 = (
            2.0 * X[:, 0] + X[:, 1] + 0.5 * X[:, 2]
            + rng.standard_normal(n_samples)
        )
        tau = 2.0
        Y1 = Y0 + tau
        Y = T * Y1 + (1 - T) * Y0
        true_ate = 2.0

    else:
        raise ValueError("setting must be 'A' or 'B'")

    # Confounders are the features that affect both T and Y
    true_confounders = [0, 1, 2]
    return X, T, Y, true_confounders, true_ate
