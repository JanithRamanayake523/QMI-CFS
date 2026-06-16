"""Data preprocessing pipeline for QMI-CFS."""
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Dict, Optional, Tuple


class QMIPreprocessor:
    """Preprocessing pipeline for quantum causal feature selection.

    Applies the following transformations in order:

    1. **Median imputation** of missing values.
    2. **Winsorization** at ``outlier_threshold * sigma`` to handle outliers.
    3. **Min-max normalization** to the interval ``[delta, pi - delta]``,
       mapping features into a range suitable for quantum encoding.
    4. **Outcome standardization** (zero mean, unit variance).

    Parameters
    ----------
    delta : float
        Lower/upper offset for normalization range. Default is 0.05.
    outlier_threshold : float
        Number of standard deviations for winsorization. Default is 3.0.

    Attributes
    ----------
    _fitted : bool
        Whether the preprocessor has been fit.
    _imputer : SimpleImputer
        Fitted median imputer.
    _x_min : np.ndarray
        Per-feature minimum values (learned from fit).
    _x_max : np.ndarray
        Per-feature maximum values (learned from fit).
    _y_mean : float
        Outcome mean (learned from fit).
    _y_std : float
        Outcome standard deviation (learned from fit).
    """

    def __init__(self, delta: float = 0.05, outlier_threshold: float = 3.0):
        self.delta = delta
        self.outlier_threshold = outlier_threshold
        self._fitted = False
        self._imputer: Optional[SimpleImputer] = None
        self._x_min: Optional[np.ndarray] = None
        self._x_max: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def _winsorize(self, X: np.ndarray) -> np.ndarray:
        """Apply winsorization column-wise.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Winsorized feature matrix.
        """
        X_out = X.copy()
        for j in range(X_out.shape[1]):
            mu, sig = np.mean(X_out[:, j]), np.std(X_out[:, j])
            if sig > 0:
                lo = mu - self.outlier_threshold * sig
                hi = mu + self.outlier_threshold * sig
                X_out[:, j] = np.clip(X_out[:, j], lo, hi)
        return X_out

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalize to [delta, pi - delta].

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Normalized feature matrix.
        """
        rng = self._x_max - self._x_min
        rng[rng == 0] = 1.0
        return self.delta + (X - self._x_min) / rng * (np.pi - 2 * self.delta)

    def fit_transform(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Fit preprocessor and transform data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariate matrix.
        T : np.ndarray of shape (n_samples,)
            Treatment assignment (binary).
        Y : np.ndarray of shape (n_samples,)
            Outcome vector.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``'X'``: normalized covariates
            - ``'T'``: treatment assignment (unchanged)
            - ``'Y'``: standardized outcomes
            - ``'y_mean'``: outcome mean (for denormalization)
            - ``'y_std'``: outcome std (for denormalization)
        """
        # Step 1: Imputation
        self._imputer = SimpleImputer(strategy='median')
        X_imp = self._imputer.fit_transform(X)

        # Step 2: Winsorization
        X_imp = self._winsorize(X_imp)

        # Step 3: Min-max normalization
        self._x_min = X_imp.min(axis=0)
        self._x_max = X_imp.max(axis=0)
        X_norm = self._normalize(X_imp)

        # Step 4: Outcome standardization
        self._y_mean = float(np.mean(Y))
        self._y_std = float(np.std(Y))
        if self._y_std == 0:
            self._y_std = 1.0
        Y_std = (Y - self._y_mean) / self._y_std

        self._fitted = True
        return {
            'X': X_norm,
            'T': T,
            'Y': Y_std,
            'y_mean': np.array([self._y_mean]),
            'y_std': np.array([self._y_std]),
        }

    def transform(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Transform new data using fitted parameters.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariate matrix.
        T : np.ndarray of shape (n_samples,)
            Treatment assignment (binary).
        Y : np.ndarray of shape (n_samples,)
            Outcome vector.

        Returns
        -------
        dict
            Dictionary with keys: 'X', 'T', 'Y', 'y_mean', 'y_std'.

        Raises
        ------
        RuntimeError
            If ``fit_transform`` has not been called prior.
        """
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor must be fit with fit_transform() before transform()."
            )

        # Imputation using fitted imputer
        X_imp = self._imputer.transform(X)

        # Winsorization
        X_imp = self._winsorize(X_imp)

        # Normalization using learned min/max
        X_norm = self._normalize(X_imp)

        # Standardization using learned statistics
        Y_std = (Y - self._y_mean) / self._y_std

        return {
            'X': X_norm,
            'T': T,
            'Y': Y_std,
            'y_mean': np.array([self._y_mean]),
            'y_std': np.array([self._y_std]),
        }

    def inverse_transform_y(self, Y_std: np.ndarray) -> np.ndarray:
        """Convert standardized outcomes back to original scale.

        Parameters
        ----------
        Y_std : np.ndarray
            Standardized outcome values.

        Returns
        -------
        np.ndarray
            Outcomes in original scale.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before inverse_transform_y().")
        return Y_std * self._y_std + self._y_mean
