#!/usr/bin/env python3
"""F1 vs. ATE-error correlation on identical simulation draws.

Experiments 1 and 2 were run on different random draws, so the main-text
correlation compares F1 and ATE error across different datasets. Experiment 3
(small-sample, LLD-style DGP) and the IHDP-inspired benchmark record F1 and
ATE error for the *same* draw, so they allow a same-draw check without
re-running any experiment.

Two analyses per condition (sample size n, or IHDP):
  * pooled:  Pearson / Spearman over all (method, replication) pairs
  * within:  Spearman across methods inside each replication, then averaged
             (removes draw-to-draw difficulty differences)

Usage:
    python scripts/same_draw_correlation.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats


def analyse(df, condition, rep_col):
    df = df.dropna(subset=['f1', 'ate_error'])
    r, p = stats.pearsonr(df['f1'], df['ate_error'])
    rho, p_rho = stats.spearmanr(df['f1'], df['ate_error'])

    within = []
    for _, g in df.groupby(rep_col):
        if g['f1'].nunique() > 1 and g['ate_error'].nunique() > 1:
            within.append(stats.spearmanr(g['f1'], g['ate_error'])[0])
    within = np.array(within)
    # One-sample Wilcoxon: is the median within-replication correlation != 0?
    p_within = stats.wilcoxon(within)[1] if len(within) > 5 else np.nan

    return {
        'condition': condition,
        'n_pairs': len(df),
        'pearson_r': r,
        'pearson_p': p,
        'R2': r ** 2,
        'spearman_rho': rho,
        'spearman_p': p_rho,
        'within_rep_mean_rho': within.mean(),
        'within_rep_frac_negative': (within < 0).mean(),
        'within_rep_wilcoxon_p': p_within,
        'n_reps': len(within),
    }


def main():
    rows = []
    exp3 = pd.read_csv('results/experiment3/experiment3_results.csv')
    for n, sub in exp3.groupby('n_samples'):
        rows.append(analyse(sub, f'Exp3 n={n}', 'replication'))
    rows.append(analyse(exp3, 'Exp3 all n (pooled)', ['n_samples', 'replication']))

    ihdp = pd.read_csv('results/ihdp/ihdp_results.csv')
    rows.append(analyse(ihdp, 'IHDP-inspired', 'seed'))
    rows.append(analyse(ihdp[ihdp['method'] != 'All-Features'],
                        'IHDP-inspired (excl. All-Features)', 'seed'))

    out = pd.DataFrame(rows)
    os.makedirs('results/same_draw', exist_ok=True)
    out.to_csv('results/same_draw/same_draw_correlation.csv', index=False)

    pd.set_option('display.width', 200)
    print(out.round(3).to_string(index=False))
    print("\nNegative rho = higher F1 goes with lower ATE error (i.e. F1 IS predictive).")
    print("Saved results/same_draw/same_draw_correlation.csv")


if __name__ == '__main__':
    main()
