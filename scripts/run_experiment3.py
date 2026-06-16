#!/usr/bin/env python3
"""Run Experiment 3: Small-Sample Analysis.

Usage:
    python scripts/run_experiment3.py --sample-sizes 100 250 500 1000 \\
        --methods all --replications 50 --output results/exp3/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qmi_cfs.data_generation import generate_fixed_ss
from qmi_cfs.feature_selection import (
    QMIFeatureSelector, MIFeatureSelector, CMIFeatureSelector,
    LASSOFeatureSelector, BorutaFeatureSelector,
    NoSelectionSelector,
)
from qmi_cfs.experiments import ExperimentRunner
from qmi_cfs.visualization import plot_experiment3_results
from qmi_cfs.statistical_tests import run_full_statistical_analysis
from qmi_cfs.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description='Experiment 3: Small-Sample Analysis')
    parser.add_argument('--sample-sizes', nargs='+', type=int, default=[100, 250, 500, 1000])
    parser.add_argument('--methods', nargs='+', default=['all'])
    parser.add_argument('--replications', type=int, default=50)
    parser.add_argument('--output', type=str, default='results/exp3')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    all_methods = {
        'QMI-CFS': lambda random_state: QMIFeatureSelector(
            random_state=random_state,
            adaptive_entanglement_neighbors=2,
            adaptive_entanglement_weight=0.2,
        ),
        'MI': MIFeatureSelector,
        'CMI': CMIFeatureSelector,
        'LASSO': LASSOFeatureSelector,
        'Boruta': BorutaFeatureSelector,
        'All-Features': NoSelectionSelector,
    }
    methods = all_methods if 'all' in args.methods else {k: v for k, v in all_methods.items() if k in args.methods}

    ensure_dir(args.output)
    runner = ExperimentRunner(results_dir=args.output, random_state=args.seed)

    print(f"Running Experiment 3: {len(args.sample_sizes)} sample sizes x {len(methods)} methods x {args.replications} reps")
    df = runner.run_experiment3(generate_fixed_ss, methods,
                                 sample_sizes=args.sample_sizes,
                                 n_replications=args.replications)
    runner.save_results(df, 'experiment3_results')

    plot_path = os.path.join(args.output, 'experiment3_small_sample_plot.pdf')
    plot_experiment3_results(df, plot_path)
    print(f"Results saved to {args.output}")

    for n in sorted(df['n_samples'].unique()):
        print(f"\n=== N={n} ===")
        sub = df[df['n_samples'] == n]
        summary = sub.groupby('method')[['f1', 'ate_error']].mean().round(4)
        print(summary)

    # Statistical significance testing per sample size
    print("\n=== Statistical Significance (Wilcoxon, Holm-Bonferroni) ===")
    for metric in ['f1', 'ate_error']:
        print(f"\n--- {metric.upper()} ---")
        stats_results = run_full_statistical_analysis(df, metric=metric, condition_cols=['n_samples'], baseline='QMI-CFS')
        wilcoxon_df = stats_results['wilcoxon']
        if not wilcoxon_df.empty:
            print(wilcoxon_df[['n_samples', 'method', 'mean_diff', 'p_value_corrected', 'significance']].round(6).to_string(index=False))
            out_path = os.path.join(args.output, f'experiment3_wilcoxon_{metric}.csv')
            wilcoxon_df.to_csv(out_path, index=False)
            print(f"Saved to {out_path}")
        else:
            print("No statistical test results.")

    # Runtime summary
    if 'runtime' in df.columns:
        print("\n=== Runtime Summary (seconds) ===")
        runtime_summary = df.groupby(['n_samples', 'method'])['runtime'].agg(['mean', 'std'])
        print(runtime_summary.round(4))


if __name__ == '__main__':
    main()
