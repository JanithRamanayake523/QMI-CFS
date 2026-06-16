#!/usr/bin/env python3
"""Run Experiment 2: Treatment Effect Estimation.

Usage:
    python scripts/run_experiment2.py --datasets all --methods all \\
        --seeds 30 --output results/exp2/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qmi_cfs.data_generation import get_all_datasets
from qmi_cfs.feature_selection import (
    QMIFeatureSelector, MIFeatureSelector, CMIFeatureSelector,
    LASSOFeatureSelector, BorutaFeatureSelector,
    NoSelectionSelector,
)
from qmi_cfs.experiments import ExperimentRunner
from qmi_cfs.visualization import plot_experiment2_results
from qmi_cfs.statistical_tests import run_full_statistical_analysis
from qmi_cfs.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description='Experiment 2: Treatment Effect Estimation')
    parser.add_argument('--datasets', nargs='+', default=['all'])
    parser.add_argument('--methods', nargs='+', default=['all'])
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--n-bootstrap', type=int, default=20,
                        help='Bootstrap replications for QMI-CFS (default 20 for speed)')
    parser.add_argument('--output', type=str, default='results/exp2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nonlinear-aipw', action='store_true',
                        help='Use Random Forest propensity and outcome models for QMI-CFS AIPW')
    args = parser.parse_args()

    all_datasets = get_all_datasets()
    datasets = all_datasets if 'all' in args.datasets else {k: v for k, v in all_datasets.items() if k in args.datasets}

    all_methods = {
        'QMI-CFS': lambda random_state: QMIFeatureSelector(
            random_state=random_state,
            adaptive_entanglement_neighbors=2,
            adaptive_entanglement_weight=0.2,
            n_bootstrap=args.n_bootstrap,
        ),
        'MI': MIFeatureSelector,
        'CMI': CMIFeatureSelector,
        'LASSO': LASSOFeatureSelector,
        'Boruta': BorutaFeatureSelector,
        'All-Features': NoSelectionSelector,
    }
    methods = all_methods if 'all' in args.methods else {k: v for k, v in all_methods.items() if k in args.methods}

    ensure_dir(args.output)
    runner = ExperimentRunner(results_dir=args.output, random_state=args.seed,
                              use_nonlinear_aipw=args.nonlinear_aipw)

    print(f"Running Experiment 2: {len(datasets)} datasets x {len(methods)} methods x {args.seeds} seeds")
    df = runner.run_experiment2(datasets, methods, n_seeds=args.seeds)
    runner.save_results(df, 'experiment2_results')

    plot_path = os.path.join(args.output, 'experiment2_ate_error_plot.pdf')
    plot_experiment2_results(df, plot_path)
    print(f"Results saved to {args.output}")

    summary = df.groupby(['dataset', 'method'])['ate_error'].agg(['mean', 'std'])
    print("\n=== Summary: Mean ATE Error ===")
    print(summary.round(4))

    # Statistical significance testing
    print("\n=== Statistical Significance (Wilcoxon, Holm-Bonferroni) ===")
    stats_results = run_full_statistical_analysis(df, metric='ate_error', condition_cols=['dataset'], baseline='QMI-CFS')
    wilcoxon_df = stats_results['wilcoxon']
    if not wilcoxon_df.empty:
        print(wilcoxon_df[['dataset', 'method', 'mean_diff', 'p_value_corrected', 'significance']].round(6).to_string(index=False))
        wilcoxon_path = os.path.join(args.output, 'experiment2_wilcoxon.csv')
        wilcoxon_df.to_csv(wilcoxon_path, index=False)
        print(f"\nStatistical test results saved to {wilcoxon_path}")
    else:
        print("No statistical test results (possibly too few valid pairs).")

    # Runtime summary
    if 'runtime' in df.columns:
        print("\n=== Runtime Summary (seconds) ===")
        runtime_summary = df.groupby(['dataset', 'method'])['runtime'].agg(['mean', 'std'])
        print(runtime_summary.round(4))


if __name__ == '__main__':
    main()
