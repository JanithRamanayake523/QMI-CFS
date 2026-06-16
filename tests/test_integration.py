"""Integration tests for the QMI-CFS package.

These tests verify that the core modules can be imported and that a minimal
end-to-end pipeline (data generation -> QMI selection -> AIPW estimation)
runs without errors.
"""

import numpy as np
import pytest

class TestImports:
    """Verify that all public modules are importable."""

    def test_import_data_generation(self):
        from qmi_cfs.data_generation import generate_lld
        assert callable(generate_lld)

    def test_import_feature_selection(self):
        from qmi_cfs.feature_selection import QMIFeatureSelector
        assert QMIFeatureSelector is not None

    def test_import_qmi_computation(self):
        from qmi_cfs.qmi_computation import QMICalculator
        assert QMICalculator is not None

    def test_import_causal_estimation(self):
        from qmi_cfs.causal_estimation import AIPWEstimator
        assert AIPWEstimator is not None


class TestFullPipeline:
    """End-to-end pipeline smoke test."""

    def test_full_pipeline_small(self):
        from qmi_cfs.data_generation import generate_lld
        from qmi_cfs.feature_selection import QMIFeatureSelector
        from qmi_cfs.causal_estimation import AIPWEstimator

        X, T, Y, true_confounders, true_ate = generate_lld(
            n_samples=200, n_features=20, seed=42
        )

        selector = QMIFeatureSelector(
            alpha=0.4, beta=0.4, gamma=0.2,
            n_bootstrap=20, stability_threshold=0.6, random_state=42
        )
        selector.fit(X, T, Y)
        selected = selector.get_selected_features()

        assert isinstance(selected, (list, np.ndarray))
        assert len(selected) <= X.shape[1]
        assert len(selected) >= 1

        estimator = AIPWEstimator(random_state=42)
        estimator.fit(X[:, selected], T, Y)
        ate_estimate = estimator.estimate_ate(X[:, selected], T, Y)

        assert np.isfinite(ate_estimate)
        assert abs(ate_estimate - true_ate) < 2.0  # loose sanity bound
