"""Feature selection methods for causal inference.

Includes QMI-CFS (quantum mutual information) and classical baselines:
MI, CMI, LASSO, Boruta, Random Forest importance.
"""
import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class BaseSelector(ABC):
    """Abstract base class for feature selectors."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'BaseSelector':
        """Fit the selector on data."""
        ...
    
    @abstractmethod
    def get_selected_features(self) -> List[int]:
        """Return list of selected feature indices."""
        ...
    
    @abstractmethod
    def get_relevance_scores(self) -> np.ndarray:
        """Return relevance score for each feature."""
        ...


class QMIFeatureSelector(BaseSelector):
    """QMI-CFS: Quantum Mutual Information Causal Feature Selection.
    
    Uses quantum mutual information computed from density matrices to select
    confounders for treatment effect estimation.
    
    Algorithm:
    1. Compute QMI: I_Q(X_j;T), I_Q(X_j;Y), I_Q(X_j;T|Y) for each feature
    2. Composite score: R(X_j) = alpha*I_Q(X_j;T) + beta*I_Q(X_j;Y) + gamma*I_Q(X_j;T|Y)
    3. Quantile-based initial screening
    4. Bootstrap stability screening
    5. Frequency thresholding for final selection
    """
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.4, gamma: float = 0.2,
                 quantile_threshold: float = 0.5, n_bootstrap: int = 100,
                 stability_threshold: float = 0.6, use_entanglement: bool = False,
                 adaptive_entanglement_neighbors: int = 0,
                 adaptive_entanglement_weight: float = 0.3,
                 delta: float = 0.05, random_state: Optional[int] = None):
        if abs(alpha + beta + gamma - 1.0) > 0.01:
            warnings.warn(f"Weights sum to {alpha+beta+gamma} != 1.0. Normalizing.")
            total = alpha + beta + gamma
            alpha, beta, gamma = alpha/total, beta/total, gamma/total
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.quantile_threshold = quantile_threshold
        self.n_bootstrap = n_bootstrap
        self.stability_threshold = stability_threshold
        self.use_entanglement = use_entanglement or (adaptive_entanglement_neighbors > 0)
        self.adaptive_entanglement_neighbors = adaptive_entanglement_neighbors
        self.adaptive_entanglement_weight = adaptive_entanglement_weight
        self.delta = delta
        self.random_state = random_state
        self._scores: Optional[np.ndarray] = None
        self._selected: Optional[List[int]] = None
        self._frequencies: Optional[np.ndarray] = None
        self._relevance_scores: Optional[np.ndarray] = None
    
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Scale features to [delta, pi - delta]."""
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        x_range = X_max - X_min
        x_range[x_range == 0] = 1.0
        return self.delta + (X - X_min) / x_range * (np.pi - 2 * self.delta)

    def _compute_relevance(
        self,
        calc: 'QMICalculator',
        X_proc: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        pair_score: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute per-feature relevance scores."""
        qmi_scores = calc.compute_all_qmi_scores(X_proc, T, Y)
        relevance = (self.alpha * qmi_scores['qmi_feature_treatment'] +
                     self.beta * qmi_scores['qmi_feature_outcome'] +
                     self.gamma * qmi_scores['conditional_qmi'])

        if pair_score is not None:
            relevance = (
                (1.0 - self.adaptive_entanglement_weight) * relevance
                + self.adaptive_entanglement_weight * pair_score
            )

        return relevance

    def _compute_pair_score(
        self,
        calc: 'QMICalculator',
        X_proc: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute the fast pairwise interaction score from entangled pair QMI.

        This is computed once on the full data and reused during bootstrap
        stability screening to keep runtime manageable.
        """
        n_features = X_proc.shape[1]
        if self.adaptive_entanglement_neighbors <= 0 or n_features <= 1:
            return None

        pair_qmi = calc.compute_pairwise_qmi_matrix(X_proc)
        pair_score = np.zeros(n_features, dtype=np.float64)
        for j in range(n_features):
            scores = pair_qmi[j].copy()
            scores[j] = -1.0
            top_k = np.argsort(scores)[-self.adaptive_entanglement_neighbors:]
            if len(top_k) == 0:
                continue
            pair_score[j] = float(np.max(pair_qmi[j, top_k]))
        return pair_score

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'QMIFeatureSelector':
        from qmi_cfs.quantum_encoding import AngleEncoder
        from qmi_cfs.qmi_computation import QMICalculator

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        X_proc = self._preprocess(X)

        encoder = AngleEncoder(n_features=n_features,
                               use_entanglement=self.use_entanglement)
        calc = QMICalculator(encoder)

        # Step 1: Compute pairwise interaction score once on full data
        pair_score = self._compute_pair_score(calc, X_proc)

        # Step 2: Compute relevance scores
        relevance = self._compute_relevance(calc, X_proc, T, Y, pair_score)
        self._relevance_scores = relevance

        # Step 3: Quantile-based initial screening
        q_thresh = np.quantile(relevance, self.quantile_threshold)
        initial_candidates = np.where(relevance >= q_thresh)[0].tolist()
        if len(initial_candidates) == 0:
            initial_candidates = [int(np.argmax(relevance))]

        # Step 4: Bootstrap stability screening
        selection_counts = np.zeros(n_features)
        for b in range(self.n_bootstrap):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_b, T_b, Y_b = X[idx], T[idx], Y[idx]
            X_b_proc = self._preprocess(X_b)
            try:
                encoder_b = AngleEncoder(n_features=n_features,
                                         use_entanglement=self.use_entanglement)
                calc_b = QMICalculator(encoder_b)
                rel_b = self._compute_relevance(
                    calc_b, X_b_proc, T_b, Y_b, pair_score
                )
                q_b = np.quantile(rel_b, self.quantile_threshold)
                sel_b = np.where(rel_b >= q_b)[0]
                for s in sel_b:
                    selection_counts[s] += 1
            except Exception:
                continue

        # Step 5: Frequency thresholding
        if self.n_bootstrap > 0:
            freq = selection_counts / self.n_bootstrap
            self._frequencies = freq
            final_selected = np.where(freq >= self.stability_threshold)[0].tolist()
        else:
            self._frequencies = np.zeros(n_features)
            final_selected = initial_candidates

        if len(final_selected) == 0:
            final_selected = [int(np.argmax(relevance))]

        self._selected = sorted(final_selected)
        self._scores = relevance
        return self
    
    def get_selected_features(self) -> List[int]:
        return self._selected if self._selected is not None else []
    
    def get_relevance_scores(self) -> np.ndarray:
        return self._relevance_scores if self._relevance_scores is not None else np.array([])
    
    def get_selection_frequencies(self) -> np.ndarray:
        return self._frequencies if self._frequencies is not None else np.array([])


class MIFeatureSelector(BaseSelector):
    """Classical Mutual Information feature selection.
    
    Score: min(MI(X_j; T), MI(X_j; Y)) for confounder identification.
    Selects top-k features by score.
    """
    
    def __init__(self, n_features: Optional[int] = None, n_neighbors: int = 3,
                 random_state: Optional[int] = None):
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._selected: Optional[List[int]] = None
        self._scores: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'MIFeatureSelector':
        mi_t = mutual_info_classif(X, T, discrete_features=False,
                                    n_neighbors=self.n_neighbors,
                                    random_state=self.random_state)
        mi_y = mutual_info_regression(X, Y,
                                       n_neighbors=self.n_neighbors,
                                       random_state=self.random_state)
        scores = np.minimum(mi_t, mi_y)
        self._scores = scores
        k = self.n_features if self.n_features is not None else max(1, X.shape[1] // 4)
        top_k = np.argsort(scores)[-k:]
        self._selected = sorted(top_k.tolist())
        return self
    
    def get_selected_features(self) -> List[int]:
        return self._selected if self._selected is not None else []
    
    def get_relevance_scores(self) -> np.ndarray:
        return self._scores if self._scores is not None else np.array([])


class CMIFeatureSelector(BaseSelector):
    """Conditional Mutual Information feature selection.
    
    For binary T: I(X; Y | T) = P(T=0)*I(X;Y|T=0) + P(T=1)*I(X;Y|T=1)
    """
    
    def __init__(self, n_features: Optional[int] = None, n_neighbors: int = 3,
                 random_state: Optional[int] = None):
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._selected: Optional[List[int]] = None
        self._scores: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'CMIFeatureSelector':
        scores = np.zeros(X.shape[1])
        for t_val in [0, 1]:
            mask = T == t_val
            if mask.sum() < self.n_neighbors + 1:
                continue
            weight = mask.mean()
            mi_y = mutual_info_regression(X[mask], Y[mask],
                                           n_neighbors=self.n_neighbors,
                                           random_state=self.random_state)
            scores += weight * mi_y
        self._scores = scores
        k = self.n_features if self.n_features is not None else max(1, X.shape[1] // 4)
        top_k = np.argsort(scores)[-k:]
        self._selected = sorted(top_k.tolist())
        return self
    
    def get_selected_features(self) -> List[int]:
        return self._selected if self._selected is not None else []
    
    def get_relevance_scores(self) -> np.ndarray:
        return self._scores if self._scores is not None else np.array([])


class LASSOFeatureSelector(BaseSelector):
    """Post-double selection LASSO (Belloni et al. 2014).
    
    1. L1-logistic for T
    2. Lasso for Y
    3. Union of selected features
    """
    
    def __init__(self, n_alphas: int = 100, cv: int = 5, max_iter: int = 5000,
                 random_state: Optional[int] = None):
        self.n_alphas = n_alphas
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        self._selected: Optional[List[int]] = None
        self._scores: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'LASSOFeatureSelector':
        cv = 3 if len(X) < 250 else self.cv
        # L1-logistic for T
        clf = LogisticRegressionCV(Cs=self.n_alphas, cv=cv, penalty='l1',
                                    solver='saga', max_iter=self.max_iter,
                                    random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, T)
        t_selected = np.where(clf.coef_.flatten() != 0)[0].tolist()
        # Lasso for Y
        reg = LassoCV(n_alphas=self.n_alphas, cv=cv, max_iter=self.max_iter,
                      random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X, Y)
        y_selected = np.where(reg.coef_ != 0)[0].tolist()
        self._selected = sorted(list(set(t_selected) | set(y_selected)))
        self._scores = np.abs(reg.coef_)
        return self
    
    def get_selected_features(self) -> List[int]:
        return self._selected if self._selected is not None else []
    
    def get_relevance_scores(self) -> np.ndarray:
        return self._scores if self._scores is not None else np.array([])


class BorutaFeatureSelector(BaseSelector):
    """Boruta: shadow feature comparison with Random Forest.
    
    Run separately for T and Y, take intersection of confirmed features.
    Gracefully handles missing BorutaPy dependency.
    """
    
    def __init__(self, max_iter: int = 100, n_estimators: int = 500,
                 max_depth: int = 5, random_state: Optional[int] = None):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._selected: Optional[List[int]] = None
        self._scores: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'BorutaFeatureSelector':
        try:
            from boruta import BorutaPy
        except ImportError:
            warnings.warn("BorutaPy not installed. Using MI as fallback.")
            fallback = MIFeatureSelector(random_state=self.random_state)
            fallback.fit(X, T, Y)
            self._selected = fallback.get_selected_features()
            self._scores = fallback.get_relevance_scores()
            return self
        
        max_depth = 3 if len(X) < 250 else self.max_depth
        
        # Boruta for T
        rf_t = RandomForestClassifier(n_estimators=self.n_estimators,
                                       max_depth=max_depth, n_jobs=-1,
                                       random_state=self.random_state)
        boruta_t = BorutaPy(rf_t, n_estimators='auto', max_iter=self.max_iter,
                            random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boruta_t.fit(X, T)
        t_confirmed = np.where(boruta_t.support_)[0]
        
        # Boruta for Y
        rf_y = RandomForestRegressor(n_estimators=self.n_estimators,
                                      max_depth=max_depth, n_jobs=-1,
                                      random_state=self.random_state)
        boruta_y = BorutaPy(rf_y, n_estimators='auto', max_iter=self.max_iter,
                             random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boruta_y.fit(X, Y)
        y_confirmed = np.where(boruta_y.support_)[0]
        
        self._selected = sorted(np.intersect1d(t_confirmed, y_confirmed).tolist())
        if len(self._selected) == 0:
            self._selected = sorted(t_confirmed.tolist()) if len(t_confirmed) > 0 else [0]
        self._scores = np.zeros(X.shape[1])
        return self
    
    def get_selected_features(self) -> List[int]:
        return self._selected if self._selected is not None else []
    
    def get_relevance_scores(self) -> np.ndarray:
        return self._scores if self._scores is not None else np.array([])


class RFFeatureSelector(BaseSelector):
    """Random Forest feature importance.
    
    Geometric mean of normalized importances from T and Y models.
    """
    
    def __init__(self, n_estimators: int = 500, max_depth: int = 5,
                 top_k: Optional[int] = None, threshold_pct: int = 10,
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.top_k = top_k
        self.threshold_pct = threshold_pct
        self.random_state = random_state
        self._selected: Optional[List[int]] = None
        self._scores: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'RFFeatureSelector':
        max_depth = 3 if len(X) < 250 else self.max_depth
        
        rf_t = RandomForestClassifier(n_estimators=self.n_estimators,
                                       max_depth=max_depth, n_jobs=-1,
                                       random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_t.fit(X, T)
        imp_t = rf_t.feature_importances_
        
        rf_y = RandomForestRegressor(n_estimators=self.n_estimators,
                                      max_depth=max_depth, n_jobs=-1,
                                      random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_y.fit(X, Y)
        imp_y = rf_y.feature_importances_
        
        imp_t_norm = imp_t / (imp_t.sum() + 1e-12)
        imp_y_norm = imp_y / (imp_y.sum() + 1e-12)
        scores = np.sqrt(imp_t_norm * imp_y_norm)
        self._scores = scores
        
        if self.top_k is not None:
            selected = np.argsort(scores)[-self.top_k:].tolist()
        else:
            threshold = np.percentile(scores, 100 - self.threshold_pct)
            selected = np.where(scores >= threshold)[0].tolist()
        self._selected = sorted(selected)
        return self
    
    def get_selected_features(self) -> List[int]:
        return self._selected if self._selected is not None else []
    
    def get_relevance_scores(self) -> np.ndarray:
        return self._scores if self._scores is not None else np.array([])


class NoSelectionSelector(BaseSelector):
    """Baseline that selects all features (no feature selection).
    
    Used to demonstrate that feature selection itself improves downstream
    treatment effect estimation compared to using all covariates.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        self._selected: Optional[List[int]] = None
        self._scores: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'NoSelectionSelector':
        self._selected = list(range(X.shape[1]))
        self._scores = np.ones(X.shape[1])
        return self
    
    def get_selected_features(self) -> List[int]:
        return self._selected if self._selected is not None else []
    
    def get_relevance_scores(self) -> np.ndarray:
        return self._scores if self._scores is not None else np.array([])


# ---------------------------------------------------------------------------
# HSIC-based feature selection baselines
# ---------------------------------------------------------------------------

def _median_bandwidth(x: np.ndarray) -> float:
    """Median heuristic bandwidth: gamma = 1 / (2 * median(pairwise_distance^2))."""
    from scipy.spatial.distance import pdist
    x = np.asarray(x).reshape(-1, 1)
    sq_dists = pdist(x, 'sqeuclidean')
    if len(sq_dists) == 0:
        return 1.0
    med_sq = float(np.median(sq_dists))
    if med_sq < 1e-12:
        return 1.0
    return 1.0 / (2.0 * med_sq)


def _rbf_kernel(x: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF kernel for a 1-D array x."""
    from sklearn.metrics.pairwise import rbf_kernel
    return rbf_kernel(x.reshape(-1, 1), gamma=gamma)


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix in feature space: H K H."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic_score(K: np.ndarray, L: np.ndarray) -> float:
    """Compute HSIC(K, L) = trace(H K H * H L H) / (n-1)^2."""
    n = K.shape[0]
    if n < 2:
        return 0.0
    Kc = _center_kernel(K)
    Lc = _center_kernel(L)
    return float(np.trace(Kc @ Lc) / ((n - 1) ** 2))


def _hsic_1d(x: np.ndarray, y: np.ndarray, gamma_x: float, gamma_y: float) -> float:
    """HSIC between two 1-D variables."""
    K = _rbf_kernel(x, gamma_x)
    L = _rbf_kernel(y, gamma_y)
    return _hsic_score(K, L)


class HSICFeatureSelector(BaseSelector):
    """HSIC feature selection with RBF kernel.

    Selects features with strongest dependence on both treatment and outcome.
    Bandwidth can be set via median heuristic, cross-validation, or a fixed gamma.

    Parameters
    ----------
    n_features : int or None
        Number of features to select. If None, selects top quarter.
    bandwidth : {'median', 'cv'} or float
        Bandwidth strategy for the RBF kernel.
    random_state : int or None
        Random seed (used by CV splits).
    """

    def __init__(self, n_features: Optional[int] = None,
                 bandwidth: str = 'median',
                 random_state: Optional[int] = None):
        self.n_features = n_features
        self.bandwidth = bandwidth
        self.random_state = random_state
        self._selected: Optional[List[int]] = None
        self._scores: Optional[np.ndarray] = None

    def _compute_scores(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray,
                        gamma: float, gamma_t: Optional[float] = None,
                        gamma_y: Optional[float] = None) -> np.ndarray:
        """Per-feature min(HSIC(X_j;T), HSIC(X_j;Y)) for a fixed bandwidth."""
        n_features = X.shape[1]
        scores = np.zeros(n_features)
        if gamma_t is None:
            gamma_t = _median_bandwidth(T)
        if gamma_y is None:
            gamma_y = _median_bandwidth(Y)
        for j in range(n_features):
            gamma_j = gamma
            if self.bandwidth == 'median':
                gamma_j = _median_bandwidth(X[:, j])
            hsic_t = _hsic_1d(X[:, j], T, gamma_j, gamma_t)
            hsic_y = _hsic_1d(X[:, j], Y, gamma_j, gamma_y)
            scores[j] = min(hsic_t, hsic_y)
        return scores

    def _cv_bandwidth(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
        """Select bandwidth by maximizing top-k HSIC score over a coarse grid.

        This avoids expensive nested cross-validation. We search a small grid
        around the median heuristic and pick the bandwidth that yields the
        largest mean HSIC score among the top-k features. To keep the search
        fast, bandwidth selection uses a random subsample (default 200).
        """
        n_samples = X.shape[0]
        n_cv = min(200, n_samples)
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n_samples, size=n_cv, replace=False)
        X_cv, T_cv, Y_cv = X[idx], T[idx], Y[idx]

        base_gamma = _median_bandwidth(Y_cv)
        gamma_t = _median_bandwidth(T_cv)
        gamma_y = _median_bandwidth(Y_cv)
        # Coarse multiplicative grid around the median heuristic
        grid = np.array([0.5, 1.0, 2.0]) * base_gamma

        n_features = X.shape[1]
        k = self.n_features if self.n_features is not None else max(1, n_features // 4)

        best_score = -np.inf
        best_gamma = base_gamma
        for gamma in grid:
            scores = self._compute_scores(X_cv, T_cv, Y_cv, gamma,
                                          gamma_t=gamma_t, gamma_y=gamma_y)
            top_k_scores = np.sort(scores)[-k:]
            mean_top_score = float(np.mean(top_k_scores))
            if mean_top_score > best_score:
                best_score = mean_top_score
                best_gamma = float(gamma)

        return best_gamma

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'HSICFeatureSelector':
        n_features = X.shape[1]

        if isinstance(self.bandwidth, (int, float)):
            gamma = float(self.bandwidth)
            gamma_t = _median_bandwidth(T)
            gamma_y = _median_bandwidth(Y)
        elif self.bandwidth == 'cv':
            gamma = self._cv_bandwidth(X, T, Y)
            gamma_t = _median_bandwidth(T)
            gamma_y = _median_bandwidth(Y)
        else:  # median
            gamma = _median_bandwidth(Y)
            gamma_t = _median_bandwidth(T)
            gamma_y = gamma

        scores = self._compute_scores(X, T, Y, gamma, gamma_t=gamma_t, gamma_y=gamma_y)
        self._scores = scores

        k = self.n_features if self.n_features is not None else max(1, n_features // 4)
        top_k = np.argsort(scores)[-k:]
        self._selected = sorted(top_k.tolist())
        return self

    def get_selected_features(self) -> List[int]:
        return self._selected if self._selected is not None else []

    def get_relevance_scores(self) -> np.ndarray:
        return self._scores if self._scores is not None else np.array([])


class HSICLassoFeatureSelector(BaseSelector):
    """HSIC-Lasso feature selection (Yamada et al., 2014 style).

    Solves a Lasso regression where the response is the vectorized centered
    kernel of a combined treatment-outcome target and the design matrix
    contains vectorized centered kernels of each feature. Features with
    non-zero coefficients are selected.

    For causal feature selection, the target kernel is the Hadamard product
    of the centered treatment and outcome kernels, so selected features are
    those whose kernel matrices jointly explain the combined treatment and
    outcome dependency structure.

    Parameters
    ----------
    n_features : int or None
        Number of features to select. If None, selects top quarter by score.
    alpha : float or None
        Lasso penalty. If None, uses BIC on a path of alphas.
    max_iter : int
        Maximum iterations for Lasso.
    random_state : int or None
    """

    def __init__(self, n_features: Optional[int] = None,
                 alpha: Optional[float] = None,
                 max_iter: int = 5000,
                 random_state: Optional[int] = None):
        self.n_features = n_features
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self._selected: Optional[List[int]] = None
        self._scores: Optional[np.ndarray] = None

    def _fit_path(self, Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit Lasso along a path of alphas and return best coefficients by BIC."""
        from sklearn.linear_model import lasso_path

        n, p = Phi.shape
        # Alpha max is the smallest value that makes all coefficients zero
        alpha_max = np.max(np.abs(Phi.T @ y)) / n
        if alpha_max <= 0 or not np.isfinite(alpha_max):
            return np.zeros(p)

        # Geometrically decreasing path
        alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max * 1e-4), 50)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, coefs, _ = lasso_path(Phi, y, alphas=alphas, max_iter=self.max_iter,
                                     random_state=self.random_state)

        # BIC = n * log(MSE) + k * log(n)
        best_bic = np.inf
        best_coef = coefs[:, -1]
        for i in range(coefs.shape[1]):
            beta = coefs[:, i]
            resid = y - Phi @ beta
            mse = np.mean(resid ** 2)
            k = np.sum(np.abs(beta) > 1e-10)
            # Use effective sample size sqrt(n^2) = n for BIC penalty
            bic = n * np.log(mse + 1e-12) + k * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_coef = beta

        return best_coef

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'HSICLassoFeatureSelector':
        from sklearn.linear_model import Lasso

        n_samples, n_features = X.shape

        # Centered kernels for T and Y
        K_T = _center_kernel(_rbf_kernel(T, _median_bandwidth(T)))
        K_Y = _center_kernel(_rbf_kernel(Y, _median_bandwidth(Y)))
        # Combined target kernel captures dependence on both T and Y
        L = _center_kernel(K_T * K_Y)

        # Build design matrix: each column is vec(centered kernel of feature j)
        Phi = np.zeros((n_samples * n_samples, n_features))
        for j in range(n_features):
            K_j = _center_kernel(_rbf_kernel(X[:, j], _median_bandwidth(X[:, j])))
            Phi[:, j] = K_j.ravel()

        # Normalize columns to unit norm for stable Lasso
        norms = np.linalg.norm(Phi, axis=0)
        norms[norms < 1e-12] = 1.0
        Phi_norm = Phi / norms

        y = L.ravel()

        # Fit Lasso
        if self.alpha is None:
            coef = self._fit_path(Phi_norm, y)
        else:
            model = Lasso(alpha=self.alpha, max_iter=self.max_iter,
                          random_state=self.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Phi_norm, y)
            coef = model.coef_

        coef = coef / norms  # Rescale coefficients to original scale
        self._scores = np.abs(coef)

        k = self.n_features if self.n_features is not None else max(1, n_features // 4)
        top_k = np.argsort(self._scores)[-k:]
        self._selected = sorted(top_k.tolist())
        if len(self._selected) == 0:
            self._selected = [int(np.argmax(self._scores))]

        return self

    def get_selected_features(self) -> List[int]:
        return self._selected if self._selected is not None else []

    def get_relevance_scores(self) -> np.ndarray:
        return self._scores if self._scores is not None else np.array([])
