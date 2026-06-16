#!/usr/bin/env python3
"""Run Experiment 1: Feature Selection Accuracy.

Usage:
    python scripts/run_experiment1.py --datasets LLD NLC HDS CC WC \\
        --methods QMI-CFS MI CMI LASSO Boruta --replications 50 \\
        --output results/exp1/
"""
import argparse
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qmi_cfs.data_generation import get_all_datasets
from qmi_cfs.feature_selection import (
    QMIFeatureSelector, MIFeatureSelector, CMIFeatureSelector,
    LASSOFeatureSelector, BorutaFeatureSelector,
    NoSelectionSelector,
)
from qmi_cfs.experiments import ExperimentRunner
from qmi_cfs.visualization import plot_experiment1_results
from qmi_cfs.statistical_tests import run_full_statistical_analysis
from qmi_cfs.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Feature Selection Accuracy')
    parser.add_argument('--datasets', nargs='+', default=['all'],
                        help='Dataset names or "all"')
    parser.add_argument('--methods', nargs='+', default=['all'],
                        help='Method names or "all"')
    parser.add_argument('--replications', type=int, default=50)
    parser.add_argument('--n-bootstrap', type=int, default=20,
                        help='Bootstrap replications for QMI-CFS (default 20 for speed)')
    parser.add_argument('--output', type=str, default='results/exp1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    all_datasets = get_all_datasets()
    if 'all' in args.datasets:
        datasets = all_datasets
    else:
        datasets = {k: v for k, v in all_datasets.items() if k in args.datasets}

    all_methods = {
        'QMI-CFS': lambda random_state: QMIFeatureSelector(
            random_state=random_state,
            adaptive_entanglement_neighbors=2,
            adaptive_entanglement_weight=0.2,
            n_bootstrap=args.n_bootstrap,
        ),
        'QMI-CFS-Classic': lambda random_state: QMIFeatureSelector(
            random_state=random_state,
            adaptive_entanglement_neighbors=0,
            n_bootstrap=args.n_bootstrap,
        ),
        'MI': MIFeatureSelector,
        'CMI': CMIFeatureSelector,
        'LASSO': LASSOFeatureSelector,
        'Boruta': BorutaFeatureSelector,
    }
    if 'all' in args.methods:
        methods = all_methods
    else:
        methods = {k: v for k, v in all_methods.items() if k in args.methods}

    ensure_dir(args.output)
    runner = ExperimentRunner(results_dir=args.output, random_state=args.seed)

    print(f"Running Experiment 1: {len(datasets)} datasets x {len(methods)} methods x {args.replications} reps")
    df = runner.run_experiment1(datasets, methods, n_replications=args.replications)
    runner.save_results(df, 'experiment1_results')

    plot_path = os.path.join(args.output, 'experiment1_f1_plot.pdf')
    plot_experiment1_results(df, plot_path)
    print(f"Results saved to {args.output}")

    # Summary
    summary = df.groupby(['dataset', 'method'])['f1'].agg(['mean', 'std'])
    print("\n=== Summary: Mean F1 Score ===")
    print(summary.round(4))

    # Statistical significance testing
    print("\n=== Statistical Significance (Paired t-test, Holm-Bonferroni) ===")
    stats_results = run_full_statistical_analysis(df, metric='f1', condition_cols=['dataset'], baseline='QMI-CFS')
    ttest_df = stats_results['ttest']
    if not ttest_df.empty:
        print(ttest_df[['dataset', 'method', 'mean_diff', 'p_value_corrected', 'significance']].round(6).to_string(index=False))
        # Save statistical test results
        ttest_path = os.path.join(args.output, 'experiment1_ttest.csv')
        ttest_df.to_csv(ttest_path, index=False)
        print(f"\nStatistical test results saved to {ttest_path}")
    else:
        print("No statistical test results (possibly too few valid pairs).")

    # Runtime summary
    if 'runtime' in df.columns:
        print("\n=== Runtime Summary (seconds) ===")
        runtime_summary = df.groupby(['dataset', 'method'])['runtime'].agg(['mean', 'std'])
        print(runtime_summary.round(4))


if __name__ == '__main__':
    main()
