#!/usr/bin/env python3
"""Run Ablation Study for QMI-CFS.

Evaluates QMI-CFS variants to isolate the contribution of each component:
1. Full QMI-CFS (default configuration)
2. No Bootstrap (quantile thresholding only)
3. Conservative Weights (equal weighting)
4. Precision Weights (emphasize outcome prediction)
5. No Entanglement (single-qubit density matrices only)

Usage:
    python scripts/run_ablation_study.py --datasets LLD NLC --replications 50 --output results/ablation/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qmi_cfs.data_generation import get_all_datasets
from qmi_cfs.feature_selection import QMIFeatureSelector
from qmi_cfs.experiments import ExperimentRunner
from qmi_cfs.statistical_tests import run_full_statistical_analysis
from qmi_cfs.utils import ensure_dir


def make_ablation_methods(n_bootstrap=100):
    """Return dict of ablation variants as factory functions."""
    return {
        'QMI-CFS (Full)': lambda random_state: QMIFeatureSelector(
            alpha=0.4, beta=0.4, gamma=0.2,
            n_bootstrap=n_bootstrap, stability_threshold=0.6,
            adaptive_entanglement_neighbors=2,
            adaptive_entanglement_weight=0.2,
            random_state=random_state
        ),
        'QMI-CFS (No Bootstrap)': lambda random_state: QMIFeatureSelector(
            alpha=0.4, beta=0.4, gamma=0.2,
            n_bootstrap=0, stability_threshold=0.0,
            adaptive_entanglement_neighbors=2,
            adaptive_entanglement_weight=0.2,
            random_state=random_state
        ),
        'QMI-CFS (Conservative)': lambda random_state: QMIFeatureSelector(
            alpha=1.0/3, beta=1.0/3, gamma=1.0/3,
            n_bootstrap=n_bootstrap, stability_threshold=0.6,
            adaptive_entanglement_neighbors=2,
            adaptive_entanglement_weight=0.2,
            random_state=random_state
        ),
        'QMI-CFS (Precision)': lambda random_state: QMIFeatureSelector(
            alpha=0.2, beta=0.6, gamma=0.2,
            n_bootstrap=n_bootstrap, stability_threshold=0.6,
            adaptive_entanglement_neighbors=2,
            adaptive_entanglement_weight=0.2,
            random_state=random_state
        ),
        'QMI-CFS (No Pairwise)': lambda random_state: QMIFeatureSelector(
            alpha=0.4, beta=0.4, gamma=0.2,
            n_bootstrap=n_bootstrap, stability_threshold=0.6,
            adaptive_entanglement_neighbors=0,
            random_state=random_state
        ),
    }


def main():
    parser = argparse.ArgumentParser(description='QMI-CFS Ablation Study')
    parser.add_argument('--datasets', nargs='+', default=['all'])
    parser.add_argument('--replications', type=int, default=50)
    parser.add_argument('--n-bootstrap', type=int, default=100,
                        help='Bootstrap replications for QMI-CFS ablation variants (default 100)')
    parser.add_argument('--output', type=str, default='results/ablation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    all_datasets = get_all_datasets()
    if 'all' in args.datasets:
        datasets = all_datasets
    else:
        datasets = {k: v for k, v in all_datasets.items() if k in args.datasets}

    methods = make_ablation_methods(n_bootstrap=args.n_bootstrap)

    ensure_dir(args.output)
    runner = ExperimentRunner(results_dir=args.output, random_state=args.seed)

    print(f"Running Ablation Study: {len(datasets)} datasets x {len(methods)} variants x {args.replications} reps")
    df = runner.run_experiment1(datasets, methods, n_replications=args.replications)
    runner.save_results(df, 'ablation_results')

    print(f"Results saved to {args.output}")

    # Summary
    summary = df.groupby(['dataset', 'method'])['f1'].agg(['mean', 'std'])
    print("\n=== Ablation Summary: Mean F1 Score ===")
    print(summary.round(4))

    # Statistical tests against Full variant
    print("\n=== Statistical Significance vs. Full (Wilcoxon) ===")
    stats_results = run_full_statistical_analysis(
        df, metric='f1', condition_cols=['dataset'], baseline='QMI-CFS (Full)'
    )
    wilcoxon_df = stats_results['wilcoxon']
    if not wilcoxon_df.empty:
        print(wilcoxon_df[['dataset', 'method', 'mean_diff', 'p_value_corrected', 'significance']].round(6).to_string(index=False))
        wilcoxon_df.to_csv(os.path.join(args.output, 'ablation_wilcoxon.csv'), index=False)

    # Runtime summary
    if 'runtime' in df.columns:
        print("\n=== Runtime Summary (seconds) ===")
        runtime_summary = df.groupby(['dataset', 'method'])['runtime'].agg(['mean', 'std'])
        print(runtime_summary.round(4))


if __name__ == '__main__':
    main()
