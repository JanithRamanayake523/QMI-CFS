"""Quantum Mutual Information computation for QMI-CFS.

Computes von Neumann entropy and quantum mutual information from
density matrices constructed by AngleEncoder.
"""
import numpy as np
from typing import Dict, Optional, List, Tuple

from qmi_cfs.quantum_encoding import AngleEncoder


class VonNeumannEntropy:
    """Von Neumann entropy S(rho) = -Tr(rho log2 rho).

    Numerically stable computation via eigenvalue decomposition.
    """

    _EPS: float = 1e-12  #: floor for eigenvalues to avoid log(0)

    @staticmethod
    def compute(rho: np.ndarray) -> float:
        """Compute von Neumann entropy of a density matrix.

        Steps:
        1. Hermitian eigendecomposition (``eigvalsh``).
        2. Clip eigenvalues to ``[_EPS, 1.0]``.
        3. Evaluate ``-sum(l_i * log2(l_i))``.

        Parameters
        ----------
        rho : np.ndarray, shape (d, d)
            Hermitian positive-semidefinite matrix with trace 1.

        Returns
        -------
        float
            Von Neumann entropy in bits.
        """
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = np.clip(eigvals, VonNeumannEntropy._EPS, 1.0)
        return float(-np.sum(eigvals * np.log2(eigvals)))

    @staticmethod
    def compute_batch(rhos: List[np.ndarray]) -> List[float]:
        """Compute entropy for multiple density matrices.

        Parameters
        ----------
        rhos : list of np.ndarray
            Each array has shape (d, d).

        Returns
        -------
        list of float
            Entropy values in the same order as ``rhos``.
        """
        return [VonNeumannEntropy.compute(r) for r in rhos]


class QMICalculator:
    """Quantum Mutual Information (QMI) calculator for QMI-CFS.

    Computes QMI between features and treatment / outcome variables
    using density matrices produced by :class:`AngleEncoder`.

    Parameters
    ----------
    encoder : AngleEncoder or None, default=None
        Encoder instance.  If ``None``, a default encoder with
        ``n_features=1`` is created.
    """

    def __init__(self, encoder: Optional[AngleEncoder] = None) -> None:
        self.encoder = encoder or AngleEncoder(n_features=1)

    # ------------------------------------------------------------------
    # QMI: I_Q(F; T)
    # ------------------------------------------------------------------

    def compute_qmi(
        self, feature_values: np.ndarray, target_values: np.ndarray
    ) -> float:
        """Compute I_Q(F; T) = S(rho_F) + S(rho_T) - S(rho_FT).

        Uses the fast analytical path for single-qubit density matrices.

        Parameters
        ----------
        feature_values : np.ndarray, shape (n_samples,)
            Feature vector.
        target_values : np.ndarray, shape (n_samples,)
            Target vector (treatment or outcome).

        Returns
        -------
        float
            Quantum mutual information (non-negative).
        """
        rho_f = self.encoder.encode_single_feature(feature_values)
        rho_t = self.encoder.encode_single_feature(target_values)
        rho_ft = self.encoder.encode_feature_pair(feature_values, target_values)

        s_f = VonNeumannEntropy.compute(rho_f)
        s_t = VonNeumannEntropy.compute(rho_t)
        s_ft = VonNeumannEntropy.compute(rho_ft)

        return max(s_f + s_t - s_ft, 0.0)

    # ------------------------------------------------------------------
    # Conditional QMI: I_Q(F; T | C)
    # ------------------------------------------------------------------

    def compute_conditional_qmi(
        self,
        feat_vals: np.ndarray,
        target_vals: np.ndarray,
        condition_vals: np.ndarray,
    ) -> float:
        """Compute I_Q(F; T | C) = S(rho_FC) + S(rho_TC) - S(rho_C) - S(rho_FTC).

        Requires a 3-qubit joint density matrix for (F, T, C).

        Parameters
        ----------
        feat_vals : np.ndarray, shape (n_samples,)
            Feature vector.
        target_vals : np.ndarray, shape (n_samples,)
            Target vector.
        condition_vals : np.ndarray, shape (n_samples,)
            Conditioning variable.

        Returns
        -------
        float
            Conditional quantum mutual information (non-negative).
        """
        rho_fc = self.encoder.encode_feature_pair(feat_vals, condition_vals)
        rho_tc = self.encoder.encode_feature_pair(target_vals, condition_vals)
        rho_c = self.encoder.encode_single_feature(condition_vals)
        rho_ftc = self.encoder.encode_feature_triple(
            feat_vals, target_vals, condition_vals
        )

        s_fc = VonNeumannEntropy.compute(rho_fc)
        s_tc = VonNeumannEntropy.compute(rho_tc)
        s_c = VonNeumannEntropy.compute(rho_c)
        s_ftc = VonNeumannEntropy.compute(rho_ftc)

        return max(s_fc + s_tc - s_c - s_ftc, 0.0)

    # ------------------------------------------------------------------
    # Batch computation over all features
    # ------------------------------------------------------------------

    def compute_all_qmi_scores(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute QMI scores for every feature in ``X``.

        For each feature ``j``:
        * ``qmi_feature_treatment[j]``  = I_Q(X_j; T)
        * ``qmi_feature_outcome[j]``    = I_Q(X_j; Y)
        * ``conditional_qmi[j]``        = I_Q(X_j; T | Y)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        T : np.ndarray, shape (n_samples,)
            Treatment indicator.
        Y : np.ndarray, shape (n_samples,)
            Outcome variable.

        Returns
        -------
        dict of str -> np.ndarray
            Keys: ``'qmi_feature_treatment'``, ``'qmi_feature_outcome'``,
            ``'conditional_qmi'``.
        """
        n_features = X.shape[1]
        qmi_ft = np.zeros(n_features, dtype=np.float64)
        qmi_fy = np.zeros(n_features, dtype=np.float64)
        cond_qmi = np.zeros(n_features, dtype=np.float64)

        for j in range(n_features):
            qmi_ft[j] = self.compute_qmi(X[:, j], T)
            qmi_fy[j] = self.compute_qmi(X[:, j], Y)
            cond_qmi[j] = self.compute_conditional_qmi(X[:, j], T, Y)

        return {
            "qmi_feature_treatment": qmi_ft,
            "qmi_feature_outcome": qmi_fy,
            "conditional_qmi": cond_qmi,
        }

    def compute_pairwise_qmi_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise QMI matrix M[i, j] = I_Q(X_i; X_j).

        Uses entangled two-qubit density matrices to capture pairwise
        dependencies between features.  Single-feature entropies are
        computed once and reused so the pair loop only builds joint
        density matrices.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (n_features, n_features)
            Symmetric pairwise QMI matrix with zeros on the diagonal.
        """
        n_features = X.shape[1]
        M = np.zeros((n_features, n_features), dtype=np.float64)

        # Cache single-feature entropies to avoid redundant encoding.
        single_entropies = np.zeros(n_features, dtype=np.float64)
        for j in range(n_features):
            rho_j = self.encoder.encode_single_feature(X[:, j])
            single_entropies[j] = VonNeumannEntropy.compute(rho_j)

        for i in range(n_features):
            s_i = single_entropies[i]
            for j in range(i + 1, n_features):
                rho_ij = self.encoder.encode_feature_pair(X[:, i], X[:, j])
                s_ij = VonNeumannEntropy.compute(rho_ij)
                qmi = max(s_i + single_entropies[j] - s_ij, 0.0)
                M[i, j] = qmi
                M[j, i] = qmi
        return M

    def compute_subset_qmi(
        self,
        subset_features: List[np.ndarray],
        target_values: np.ndarray,
        entanglement_edges: Optional[List[Tuple[int, int]]] = None,
    ) -> float:
        """Compute I_Q(S; T) for a feature subset S and a target T.

        The subset features are encoded into a single entangled multi-qubit
        state and the QMI is computed with respect to the target.

        Parameters
        ----------
        subset_features : list of np.ndarray
            Each array has shape (n_samples,) and contains the values for one
            feature in the subset.
        target_values : np.ndarray, shape (n_samples,)
            Target vector (treatment or outcome).
        entanglement_edges : list of (int, int), optional
            CZ entanglement edges within the subset. If None, linear CZ
            entanglement is applied.

        Returns
        -------
        float
            Quantum mutual information I_Q(S; T).
        """
        # Marginal entropy of the subset
        rho_s = self.encoder.encode_feature_subset(
            subset_features, entanglement_edges=entanglement_edges
        )
        s_s = VonNeumannEntropy.compute(rho_s)

        # Marginal entropy of the target
        rho_t = self.encoder.encode_single_feature(target_values)
        s_t = VonNeumannEntropy.compute(rho_t)

        # Joint entropy of subset + target
        rho_st = self.encoder.encode_feature_subset(
            subset_features + [target_values],
            entanglement_edges=entanglement_edges,
        )
        s_st = VonNeumannEntropy.compute(rho_st)

        return max(s_s + s_t - s_st, 0.0)
