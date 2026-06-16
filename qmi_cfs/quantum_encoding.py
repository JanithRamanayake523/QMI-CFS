"""Quantum encoding using Qiskit for QMI-CFS.

Angle encoding with optional ZZFeatureMap-style entanglement.
Density matrices are constructed analytically via numpy for efficiency.
"""
import numpy as np
from typing import Optional, List, Tuple

try:
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


class AngleEncoder:
    """Angle encoder for QMI-CFS.

    Maps classical features to quantum states via R_Y rotation gates.
    Supports optional ZZ entanglement for capturing feature interactions.

    For single-qubit systems, density matrices are computed analytically
    without constructing individual statevectors.

    Parameters
    ----------
    n_features : int
        Number of classical features to encode.
    use_entanglement : bool, default=True
        Whether to apply CZ entangling gates between qubits.
    entanglement_type : str, default='linear'
        Entanglement topology. Only 'linear' is supported: CZ between
        adjacent qubit pairs (j, j+1).
    """

    def __init__(
        self,
        n_features: int,
        use_entanglement: bool = True,
        entanglement_type: str = "linear",
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        self.n_features = n_features
        self.use_entanglement = use_entanglement
        self.entanglement_type = entanglement_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_single_feature(self, feature_values: np.ndarray) -> np.ndarray:
        """Construct a 2x2 density matrix for a single feature.

        The quantum state is the mixed state
            rho = (1/N) sum_i |phi_i><phi_i|
        where |phi_i> = cos(x_i)|0> + sin(x_i)|1>.

        Analytically:
            a = mean(cos^2(x)),  c = mean(cos(x) * sin(x))
            rho = [[a, c], [c, 1 - a]]

        Parameters
        ----------
        feature_values : np.ndarray, shape (n_samples,)
            Feature values (angles or preprocessed data).

        Returns
        -------
        np.ndarray, shape (2, 2)
            Real symmetric density matrix with trace 1.
        """
        feature_values = np.asarray(feature_values, dtype=np.float64)
        cos_v = np.cos(feature_values)
        sin_v = np.sin(feature_values)
        a = float(np.mean(cos_v * cos_v))
        c = float(np.mean(cos_v * sin_v))
        rho = np.array([[a, c], [c, 1.0 - a]], dtype=np.float64)
        return rho

    def encode_feature_pair(
        self, feat1_vals: np.ndarray, feat2_vals: np.ndarray
    ) -> np.ndarray:
        """Construct a 4x4 joint density matrix for two features.

        Computes the true joint density matrix from data co-occurrence:
            rho = (1/N) sum_i |psi_i><psi_i|
        where |psi_i> = |phi_1(f1_i)> ⊗ |phi_2(f2_i)>.

        This captures statistical correlations through the co-occurrence
        patterns in the data, yielding non-zero QMI for correlated features.

        Parameters
        ----------
        feat1_vals, feat2_vals : np.ndarray, shape (n_samples,)
            Feature values for qubit 0 and qubit 1.

        Returns
        -------
        np.ndarray, shape (4, 4)
            Joint density matrix.
        """
        feat1_vals = np.asarray(feat1_vals, dtype=np.float64)
        feat2_vals = np.asarray(feat2_vals, dtype=np.float64)
        c1, s1 = np.cos(feat1_vals), np.sin(feat1_vals)
        c2, s2 = np.cos(feat2_vals), np.sin(feat2_vals)

        # |psi_i> = [c1*c2, c1*s2, s1*c2, s1*s2]
        # rho[p,q] = E[psi_p * psi_q]
        rho = np.zeros((4, 4), dtype=np.float64)
        comps = [c1 * c2, c1 * s2, s1 * c2, s1 * s2]
        for p in range(4):
            for q in range(4):
                rho[p, q] = float(np.mean(comps[p] * comps[q]))

        if self.use_entanglement:
            # Apply CZ entanglement: |11> gets a -1 phase
            # This modifies the density matrix to capture quantum correlations
            rho = self._encode_multi_feature_entangled([feat1_vals, feat2_vals])

        return rho

    def encode_feature_triple(
        self, f1: np.ndarray, f2: np.ndarray, f3: np.ndarray
    ) -> np.ndarray:
        """Construct an 8x8 joint density matrix for three features.

        Computes the true joint density matrix from data co-occurrence,
        capturing statistical correlations through the joint data distribution.

        Parameters
        ----------
        f1, f2, f3 : np.ndarray, shape (n_samples,)
            Feature values for qubits 0, 1, and 2.

        Returns
        -------
        np.ndarray, shape (8, 8)
            Joint density matrix.
        """
        f1 = np.asarray(f1, dtype=np.float64)
        f2 = np.asarray(f2, dtype=np.float64)
        f3 = np.asarray(f3, dtype=np.float64)
        c1, s1 = np.cos(f1), np.sin(f1)
        c2, s2 = np.cos(f2), np.sin(f2)
        c3, s3 = np.cos(f3), np.sin(f3)

        # |psi_i> components: all 8 combinations
        rho = np.zeros((8, 8), dtype=np.float64)
        comps = [
            c1 * c2 * c3, c1 * c2 * s3, c1 * s2 * c3, c1 * s2 * s3,
            s1 * c2 * c3, s1 * c2 * s3, s1 * s2 * c3, s1 * s2 * s3,
        ]
        for p in range(8):
            for q in range(8):
                rho[p, q] = float(np.mean(comps[p] * comps[q]))

        if self.use_entanglement:
            rho = self._encode_multi_feature_entangled([f1, f2, f3])

        return rho

    def encode_feature_subset(
        self, feature_list: List[np.ndarray],
        entanglement_edges: Optional[List[tuple]] = None
    ) -> np.ndarray:
        """Construct a joint density matrix for an arbitrary feature subset.

        Parameters
        ----------
        feature_list : list of np.ndarray
            Each array has shape (n_samples,) and contains feature values
            for one qubit, in the order they should appear in the register.
        entanglement_edges : list of (int, int), optional
            Pairs of qubit indices on which to apply CZ gates. If None and
            ``use_entanglement`` is True, linear CZ entanglement is applied.

        Returns
        -------
        np.ndarray, shape (2**n_qubits, 2**n_qubits)
            Joint density matrix.
        """
        feature_list = [np.asarray(f, dtype=np.float64) for f in feature_list]
        n_qubits = len(feature_list)
        dim = 2 ** n_qubits

        cs = [np.cos(f) for f in feature_list]
        sn = [np.sin(f) for f in feature_list]

        comps = np.ones((dim, len(feature_list[0])), dtype=np.float64)
        for b in range(dim):
            for j in range(n_qubits):
                if (b >> j) & 1:
                    comps[b] *= sn[j]
                else:
                    comps[b] *= cs[j]

        if self.use_entanglement and n_qubits > 1:
            if entanglement_edges is None:
                entanglement_edges = [(j, j + 1) for j in range(n_qubits - 1)]
            for (qa, qb) in entanglement_edges:
                mask = np.zeros(dim, dtype=bool)
                for b in range(dim):
                    if ((b >> qa) & 1) and ((b >> qb) & 1):
                        mask[b] = True
                comps[mask] *= -1.0

        rho = np.zeros((dim, dim), dtype=np.float64)
        for p in range(dim):
            for q in range(dim):
                rho[p, q] = float(np.mean(comps[p] * comps[q]))

        return rho

    def encode_treatment(self, T: np.ndarray) -> np.ndarray:
        """Encode a binary treatment variable.

        Maps T=0 -> angle pi/4 and T=1 -> angle pi/2.

        Parameters
        ----------
        T : np.ndarray, shape (n_samples,)
            Binary treatment indicator (0 or 1).

        Returns
        -------
        np.ndarray, shape (2, 2)
            Single-qubit density matrix.
        """
        T = np.asarray(T)
        angles = np.where(T == 0, np.pi / 4, np.pi / 2)
        return self.encode_single_feature(angles)

    def encode_outcome(self, Y: np.ndarray) -> np.ndarray:
        """Encode a preprocessed outcome variable.

        Assumes ``Y`` has already been scaled to the range
        ``[delta, pi - delta]`` for some small ``delta > 0``.

        Parameters
        ----------
        Y : np.ndarray, shape (n_samples,)
            Preprocessed outcome values.

        Returns
        -------
        np.ndarray, shape (2, 2)
            Single-qubit density matrix.
        """
        return self.encode_single_feature(Y)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_multi_feature_entangled(
        self, feature_list: List[np.ndarray]
    ) -> np.ndarray:
        """Build an entangled multi-qubit density matrix analytically.

        Computes rho = (1/N) sum_i |psi_i><psi_i| with linear CZ
        entanglement between adjacent qubits.  This is orders of
        magnitude faster than constructing individual Qiskit circuits.

        Parameters
        ----------
        feature_list : list of np.ndarray
            Each array has shape (n_samples,) and contains the feature
            values for one qubit.

        Returns
        -------
        np.ndarray, shape (2**n_qubits, 2**n_qubits)
            Joint density matrix (real-valued).
        """
        n_samples = len(feature_list[0])
        n_qubits = len(feature_list)
        dim = 2 ** n_qubits

        # Pre-compute cos/sin for all features
        cs = [np.cos(f) for f in feature_list]
        sn = [np.sin(f) for f in feature_list]

        # Build basis-state amplitude components
        # comps[b][i] = amplitude of basis state b for sample i
        comps = np.ones((dim, n_samples), dtype=np.float64)
        for b in range(dim):
            for j in range(n_qubits):
                if (b >> j) & 1:
                    comps[b] *= sn[j]
                else:
                    comps[b] *= cs[j]

        # Apply CZ phase flips: (-1) for each adjacent pair both in |1>
        if n_qubits > 1:
            for j in range(n_qubits - 1):
                mask = np.zeros(dim, dtype=bool)
                for b in range(dim):
                    if ((b >> j) & 1) and ((b >> (j + 1)) & 1):
                        mask[b] = True
                comps[mask] *= -1.0

        # Build density matrix: rho[p,q] = E[comps[p] * comps[q]]
        rho = np.zeros((dim, dim), dtype=np.float64)
        for p in range(dim):
            for q in range(dim):
                rho[p, q] = float(np.mean(comps[p] * comps[q]))

        return rho
