#!/usr/bin/env python3
"""Run IHDP Benchmark Experiment.

Evaluates all feature selection methods on the IHDP semi-synthetic dataset
(Setting A: N=747, 25 features, true ATE ~ 4.0).

Usage:
    python scripts/run_ihdp_experiment.py --methods all --seeds 30 --output results/ihdp
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from tqdm import tqdm

from qmi_cfs.data_generation import generate_ihdp
from qmi_cfs.feature_selection import (
    QMIFeatureSelector, MIFeatureSelector, CMIFeatureSelector,
    LASSOFeatureSelector, BorutaFeatureSelector,
    NoSelectionSelector,
)
from qmi_cfs.causal_estimation import AIPWEstimator
from qmi_cfs.evaluation import compute_all_fs_metrics, compute_ate_error
from qmi_cfs.statistical_tests import run_full_statistical_analysis
from qmi_cfs.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description='IHDP Benchmark Experiment')
    parser.add_argument('--methods', nargs='+', default=['all'],
                        help='Method names or "all"')
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of random seeds (replications)')
    parser.add_argument('--setting', type=str, default='A', choices=['A', 'B'],
                        help='IHDP setting (A or B)')
    parser.add_argument('--output', type=str, default='results/ihdp')
    parser.add_argument('--base-seed', type=int, default=42)
    args = parser.parse_args()

    all_methods = {
        'QMI-CFS': lambda rs: QMIFeatureSelector(n_bootstrap=100, random_state=rs),
        'MI': lambda rs: MIFeatureSelector(n_features=3, random_state=rs),
        'CMI': lambda rs: CMIFeatureSelector(n_features=3, random_state=rs),
        'LASSO': lambda rs: LASSOFeatureSelector(random_state=rs),
        'Boruta': lambda rs: BorutaFeatureSelector(random_state=rs),
        'All-Features': lambda rs: NoSelectionSelector(random_state=rs),
    }

    if 'all' in args.methods:
        methods = all_methods
    else:
        methods = {k: v for k, v in all_methods.items() if k in args.methods}

    ensure_dir(args.output)
    rows = []
    total = args.seeds * len(methods)
    pbar = tqdm(total=total, desc=f'IHDP Setting {args.setting}')

    for seed_idx in range(args.seeds):
        seed = args.base_seed + seed_idx * 100
        X, T, Y, true_confounders, true_ate = generate_ihdp(
            setting=args.setting, seed=seed
        )
        n_features = X.shape[1]

        for method_name, factory in methods.items():
            t0 = pd.Timestamp.now()
            try:
                selector = factory(seed)
                selector.fit(X, T, Y)
                selected = selector.get_selected_features()
                if len(selected) == 0:
                    selected = list(range(min(n_features, 5)))

                metrics = compute_all_fs_metrics(selected, true_confounders, n_features)

                aipw = AIPWEstimator(random_state=seed)
                aipw.fit(X[:, selected], T, Y)
                ate_hat = aipw.estimate_ate(X[:, selected], T, Y)
                ate_err = compute_ate_error(ate_hat, true_ate)

                rows.append({
                    'dataset': f'IHDP-{args.setting}',
                    'method': method_name,
                    'seed': seed_idx,
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'specificity': metrics['specificity'],
                    'ate_error': ate_err,
                    'n_selected': len(selected),
                    'true_ate': true_ate,
                })
            except Exception as e:
                rows.append({
                    'dataset': f'IHDP-{args.setting}',
                    'method': method_name,
                    'seed': seed_idx,
                    'f1': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'specificity': 0.0,
                    'ate_error': np.nan,
                    'n_selected': 0,
                    'true_ate': true_ate,
                })
            pbar.update(1)

    pbar.close()
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output, 'ihdp_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Summary
    print("\n=== Summary: Mean F1 & ATE Error ===")
    summary = df.groupby('method')[['f1', 'ate_error']].agg(['mean', 'std'])
    print(summary.round(4))

    # Statistical tests
    print("\n=== Statistical Significance (Wilcoxon, Holm-Bonferroni) ===")
    for metric in ['f1', 'ate_error']:
        print(f"\n--- {metric.upper()} ---")
        stats_results = run_full_statistical_analysis(
            df, metric=metric, baseline='QMI-CFS'
        )
        wilcoxon_df = stats_results['wilcoxon']
        if not wilcoxon_df.empty:
            print(wilcoxon_df[['method', 'mean_diff', 'p_value_corrected', 'significance']].round(6).to_string(index=False))
            out_path = os.path.join(args.output, f'ihdp_wilcoxon_{metric}.csv')
            wilcoxon_df.to_csv(out_path, index=False)
        else:
            print("No test results.")


if __name__ == '__main__':
    main()
