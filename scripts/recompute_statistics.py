#!/usr/bin/env python3
"""Recompute all significance tests from saved per-replication results.

Re-runs only the statistical analysis (paired t-tests / Wilcoxon tests with
Holm-Bonferroni correction) on the result CSVs already stored in ``results/``.
No experiments are re-run. Each statistics CSV is overwritten in place, using
the same test, metric, grouping and baseline as the original experiment script.

Usage:
    python scripts/recompute_statistics.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from qmi_cfs.statistical_tests import run_full_statistical_analysis

# (results csv, metric, condition cols, baseline, test, output csv)
JOBS = [
    ('results/experiment1/experiment1_results.csv', 'f1', ['dataset'], 'QMI-CFS',
     'ttest', 'results/experiment1/experiment1_ttest.csv'),
    ('results/experiment2/experiment2_results.csv', 'ate_error', ['dataset'], 'QMI-CFS',
     'wilcoxon', 'results/experiment2/experiment2_wilcoxon.csv'),
    ('results/experiment3/experiment3_results.csv', 'f1', ['n_samples'], 'QMI-CFS',
     'wilcoxon', 'results/experiment3/experiment3_wilcoxon_f1.csv'),
    ('results/experiment3/experiment3_results.csv', 'ate_error', ['n_samples'], 'QMI-CFS',
     'wilcoxon', 'results/experiment3/experiment3_wilcoxon_ate_error.csv'),
    ('results/ihdp/ihdp_results.csv', 'f1', None, 'QMI-CFS',
     'wilcoxon', 'results/ihdp/ihdp_wilcoxon_f1.csv'),
    ('results/ihdp/ihdp_results.csv', 'ate_error', None, 'QMI-CFS',
     'wilcoxon', 'results/ihdp/ihdp_wilcoxon_ate_error.csv'),
    ('results/ablation/ablation_results.csv', 'f1', ['dataset'], 'QMI-CFS (Full)',
     'wilcoxon', 'results/ablation/ablation_wilcoxon.csv'),
]


def main():
    for src, metric, cond, baseline, test, out in JOBS:
        df = pd.read_csv(src)
        old_rejects = None
        if os.path.exists(out):
            old_rejects = int(pd.read_csv(out)['reject'].sum())

        new = run_full_statistical_analysis(
            df, metric=metric, condition_cols=cond, baseline=baseline
        )[test]
        new.to_csv(out, index=False)

        print(f"\n=== {out} ({test}, metric={metric}) ===")
        print(f"Significant comparisons: before fix = {old_rejects}, "
              f"after fix = {int(new['reject'].sum())} of {len(new)}")
        cols = (cond or []) + ['method', 'mean_diff', 'p_value_raw',
                               'p_value_corrected', 'significance']
        print(new[cols].round(4).to_string(index=False))

    print("\nAll statistics recomputed. Next: python scripts/generate_tables.py "
          "--output results/tables_camera_ready")


if __name__ == '__main__':
    main()
