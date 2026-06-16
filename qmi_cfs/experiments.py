"""Experiment orchestration for QMI-CFS paper."""
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional
from tqdm import tqdm

from qmi_cfs.utils import ensure_dir, set_seed
from qmi_cfs.preprocessing import QMIPreprocessor
from qmi_cfs.causal_estimation import AIPWEstimator
from qmi_cfs.evaluation import compute_all_fs_metrics, compute_ate_error


class ExperimentRunner:
    """Orchestrates all three experiments for QMI-CFS evaluation."""

    def __init__(self, results_dir: str = 'results', random_state: int = 42,
                 use_nonlinear_aipw: bool = False):
        self.results_dir = ensure_dir(results_dir)
        self.random_state = random_state
        self.use_nonlinear_aipw = use_nonlinear_aipw

    def run_experiment1(self, datasets: Dict[str, Callable],
                        methods: Dict[str, object],
                        n_replications: int = 50,
                        n_features_select: Optional[int] = None) -> pd.DataFrame:
        """Experiment 1: Feature Selection Accuracy.

        Evaluates ability to identify true confounders via Precision, Recall, F1.
        """
        rows = []
        total = len(datasets) * n_replications * len(methods)
        pbar = tqdm(total=total, desc='Exp 1: Feature Selection')
        for dataset_name, gen_fn in datasets.items():
            for rep in range(n_replications):
                seed = self.random_state + rep
                X, T, Y, true_confounders, true_ate = gen_fn(seed=seed)
                n_features = X.shape[1]
                k = n_features_select if n_features_select is not None else len(true_confounders)

                for method_name, selector_cls in methods.items():
                    t0 = time.perf_counter()
                    try:
                        if callable(selector_cls):
                            selector = selector_cls(random_state=seed)
                        else:
                            selector = selector_cls
                        if hasattr(selector, 'n_features') and n_features_select is not None:
                            selector.n_features = k
                        selector.fit(X, T, Y)
                        selected = selector.get_selected_features()
                        metrics = compute_all_fs_metrics(selected, true_confounders, n_features)
                        rows.append({
                            'dataset': dataset_name,
                            'method': method_name,
                            'replication': rep,
                            **metrics,
                            'runtime': time.perf_counter() - t0,
                        })
                    except Exception as e:
                        rows.append({
                            'dataset': dataset_name,
                            'method': method_name,
                            'replication': rep,
                            'precision': np.nan, 'recall': np.nan, 'f1': np.nan,
                            'specificity': np.nan, 'n_selected': 0, 'n_true': len(true_confounders),
                            'runtime': time.perf_counter() - t0,
                        })
                    pbar.update(1)
        pbar.close()
        return pd.DataFrame(rows)

    def run_experiment2(self, datasets: Dict[str, Callable],
                        methods: Dict[str, object],
                        n_seeds: int = 30) -> pd.DataFrame:
        """Experiment 2: Treatment Effect Estimation.

        Evaluates whether selected features improve ATE estimation.
        """
        rows = []
        total = len(datasets) * n_seeds * len(methods)
        pbar = tqdm(total=total, desc='Exp 2: Treatment Effect Estimation')
        for dataset_name, gen_fn in datasets.items():
            for seed_idx in range(n_seeds):
                seed = self.random_state + seed_idx * 100
                X, T, Y, true_confounders, true_ate = gen_fn(seed=seed)

                for method_name, selector_cls in methods.items():
                    t0 = time.perf_counter()
                    try:
                        if callable(selector_cls):
                            selector = selector_cls(random_state=seed)
                        else:
                            selector = selector_cls
                        selector.fit(X, T, Y)
                        selected = selector.get_selected_features()
                        if len(selected) == 0:
                            selected = list(range(min(X.shape[1], 5)))

                        if self.use_nonlinear_aipw and method_name == 'QMI-CFS':
                            from sklearn.ensemble import (
                                RandomForestClassifier, RandomForestRegressor
                            )
                            from sklearn.calibration import CalibratedClassifierCV
                            aipw = AIPWEstimator(
                                ps_model=CalibratedClassifierCV(
                                    RandomForestClassifier(
                                        n_estimators=50, max_depth=4,
                                        random_state=seed, n_jobs=-1
                                    ),
                                    method='isotonic', cv=3
                                ),
                                outcome_model=RandomForestRegressor(
                                    n_estimators=50, max_depth=4,
                                    random_state=seed, n_jobs=-1
                                ),
                                random_state=seed,
                            )
                        else:
                            aipw = AIPWEstimator(random_state=seed)
                        aipw.fit(X[:, selected], T, Y)
                        ate_hat = aipw.estimate_ate(X[:, selected], T, Y)
                        ate_err = compute_ate_error(ate_hat, true_ate)

                        rows.append({
                            'dataset': dataset_name,
                            'method': method_name,
                            'seed': seed_idx,
                            'ate_error': ate_err,
                            'relative_ate_error': ate_err / (abs(true_ate) + 1e-10) * 100,
                            'runtime': time.perf_counter() - t0,
                        })
                    except Exception as e:
                        rows.append({
                            'dataset': dataset_name,
                            'method': method_name,
                            'seed': seed_idx,
                            'ate_error': np.nan,
                            'relative_ate_error': np.nan,
                            'runtime': time.perf_counter() - t0,
                        })
                    pbar.update(1)
        pbar.close()
        return pd.DataFrame(rows)

    def run_experiment3(self, dataset_generator: Callable,
                        methods: Dict[str, object],
                        sample_sizes: list = [100, 250, 500, 1000],
                        n_replications: int = 50) -> pd.DataFrame:
        """Experiment 3: Small-Sample Analysis.

        Evaluates performance under limited data conditions.
        """
        rows = []
        total = len(sample_sizes) * n_replications * len(methods)
        pbar = tqdm(total=total, desc='Exp 3: Small-Sample Analysis')
        for n in sample_sizes:
            for rep in range(n_replications):
                seed = 1000 * n + rep
                X, T, Y, true_confounders, true_ate = dataset_generator(n_samples=n, seed=seed)
                n_features = X.shape[1]

                for method_name, selector_cls in methods.items():
                    t0 = time.perf_counter()
                    try:
                        if callable(selector_cls):
                            selector = selector_cls(random_state=seed)
                        else:
                            selector = selector_cls
                        if hasattr(selector, 'n_features'):
                            selector.n_features = len(true_confounders)
                        selector.fit(X, T, Y)
                        selected = selector.get_selected_features()

                        metrics = compute_all_fs_metrics(selected, true_confounders, n_features)
                        f1 = metrics['f1']

                        if len(selected) == 0:
                            selected = list(range(min(n_features, 5)))

                        aipw = AIPWEstimator(random_state=seed)
                        aipw.fit(X[:, selected], T, Y)
                        ate_hat = aipw.estimate_ate(X[:, selected], T, Y)
                        ate_err = compute_ate_error(ate_hat, true_ate)

                        rows.append({
                            'n_samples': n,
                            'method': method_name,
                            'replication': rep,
                            'f1': f1,
                            'ate_error': ate_err,
                            'runtime': time.perf_counter() - t0,
                        })
                    except Exception as e:
                        rows.append({
                            'n_samples': n,
                            'method': method_name,
                            'replication': rep,
                            'f1': 0.0,
                            'ate_error': np.nan,
                            'runtime': time.perf_counter() - t0,
                        })
                    pbar.update(1)
        pbar.close()
        return pd.DataFrame(rows)

    def save_results(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame as CSV and JSON."""
        csv_path = os.path.join(self.results_dir, f'{filename}.csv')
        json_path = os.path.join(self.results_dir, f'{filename}.json')
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient='records', indent=2)
