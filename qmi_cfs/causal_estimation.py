"""Causal effect estimation using Augmented Inverse Probability Weighting (AIPW).

Doubly robust estimator that is consistent if either the propensity score model
or the outcome model (but not necessarily both) is correctly specified.
"""
import numpy as np
import warnings
from typing import Optional, Tuple

from sklearn.linear_model import LogisticRegression, Ridge


class AIPWEstimator:
    """Augmented Inverse Probability Weighting (Doubly Robust) Estimator.
    
    tau_hat = (1/N) * sum_i [
        mu_1(X_i) - mu_0(X_i)
        + T_i/e(X_i) * (Y_i - mu_1(X_i))
        - (1-T_i)/(1-e(X_i)) * (Y_i - mu_0(X_i))
    ]
    """
    
    def __init__(self,
                 ps_model=None,
                 outcome_model=None,
                 ps_clip: Tuple[float, float] = (0.05, 0.95),
                 use_firth: bool = False,
                 random_state: Optional[int] = None):
        self.ps_model = ps_model
        self.outcome_model = outcome_model
        self.ps_clip = ps_clip
        self.use_firth = use_firth
        self.random_state = random_state
        self._ps_model = None
        self._outcome_models = {}
        self._fitted = False
    
    def fit(self, X_selected: np.ndarray, T: np.ndarray,
            Y: np.ndarray) -> 'AIPWEstimator':
        """Fit propensity score and outcome models.
        
        Args:
            X_selected: Selected features, shape (n_samples, n_selected)
            T: Treatment indicator, shape (n_samples,)
            Y: Outcome, shape (n_samples,)
        """
        from sklearn.base import clone

        # Propensity score model
        if self.ps_model is None:
            self._ps_model = LogisticRegression(C=1.0, max_iter=1000,
                                                random_state=self.random_state)
        else:
            self._ps_model = clone(self.ps_model)
            if hasattr(self._ps_model, 'random_state'):
                self._ps_model.set_params(random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._ps_model.fit(X_selected, T)
        
        # Outcome models (separate per treatment arm)
        if self.outcome_model is None:
            outcome_base = Ridge()
        else:
            outcome_base = clone(self.outcome_model)
            if hasattr(outcome_base, 'random_state'):
                outcome_base.set_params(random_state=self.random_state)
        for t_val in [0, 1]:
            mask = T == t_val
            if mask.sum() < 2:
                self._outcome_models[t_val] = None
                continue
            model = clone(outcome_base)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_selected[mask], Y[mask])
            self._outcome_models[t_val] = model
        
        self._fitted = True
        return self
    
    def _get_propensity_scores(self, X: np.ndarray) -> np.ndarray:
        """Get clipped propensity scores."""
        ps = self._ps_model.predict_proba(X)[:, 1]
        return np.clip(ps, self.ps_clip[0], self.ps_clip[1])
    
    def _get_outcome_prediction(self, X: np.ndarray, t_val: int) -> np.ndarray:
        """Get outcome predictions for a treatment arm."""
        model = self._outcome_models.get(t_val)
        if model is None:
            return np.zeros(len(X))
        return model.predict(X)
    
    def predict_ate(self, X_selected: np.ndarray) -> float:
        """Compute ATE estimate using the AIPW formula."""
        raise RuntimeError("Must provide T and Y. Use estimate_ate instead.")
    
    def estimate_ate(self, X_selected: np.ndarray, T: np.ndarray,
                     Y: np.ndarray) -> float:
        """Estimate ATE from the full AIPW influence function.
        
        Returns:
            ATE estimate (float)
        """
        ps = self._get_propensity_scores(X_selected)
        mu1 = self._get_outcome_prediction(X_selected, 1)
        mu0 = self._get_outcome_prediction(X_selected, 0)
        
        # AIPW influence function
        psi = (mu1 - mu0 
               + T / ps * (Y - mu1)
               - (1 - T) / (1 - ps) * (Y - mu0))
        
        return float(np.mean(psi))
    
    def compute_influence_function(self, X: np.ndarray, T: np.ndarray,
                                    Y: np.ndarray) -> np.ndarray:
        """Compute influence function values for each observation.
        
        Returns:
            Influence function values, shape (n_samples,)
        """
        ps = self._get_propensity_scores(X)
        mu1 = self._get_outcome_prediction(X, 1)
        mu0 = self._get_outcome_prediction(X, 0)
        
        psi = (mu1 - mu0
               + T / ps * (Y - mu1)
               - (1 - T) / (1 - ps) * (Y - mu0))
        return psi
    
    def compute_confidence_interval(self, X: np.ndarray, T: np.ndarray,
                                     Y: np.ndarray,
                                     alpha: float = 0.05) -> Tuple[float, float]:
        """Compute (1-alpha) confidence interval for ATE.
        
        Returns:
            (lower_bound, upper_bound)
        """
        psi = self.compute_influence_function(X, T, Y)
        tau_hat = float(np.mean(psi))
        n = len(psi)
        se = float(np.std(psi, ddof=1) / np.sqrt(n))
        z = 1.96 if alpha == 0.05 else 2.576
        return (tau_hat - z * se, tau_hat + z * se)
