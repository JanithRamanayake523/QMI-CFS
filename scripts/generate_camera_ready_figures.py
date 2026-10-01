#!/usr/bin/env python3
"""Regenerate the camera-ready figures from saved results (no experiments are run).

Writes new files next to the originals, which are left untouched:
  paper/final_submission/figures/v6_ihdp_combined.{pdf,png}         -- all 10 methods, IHDP-inspired labels,
                                                      bars coloured by significance vs QMI-CFS
  (unused) v6_f1_vs_ate_by_dataset.{pdf,png}  -- per-DGP scatter without fitted lines

Usage:
    python scripts/generate_camera_ready_figures.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
from qmi_cfs.plot_style import set_publication_style, METHOD_COLORS
set_publication_style()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = 'paper/final_submission/figures'

# Embed TrueType instead of Type 3 fonts (Springer production requirement).
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Colours for the IHDP figure: QMI-CFS highlighted, other methods coloured by
# whether their ATE error differs significantly from QMI-CFS (Holm-corrected).
C_QMI = METHOD_COLORS['QMI-CFS']
C_TIED = '#2a9d8f'
C_DIFF = '#f7e48b'


def _save(fig, name):
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


def plot_ihdp_combined():
    df = pd.read_csv('results/ihdp/ihdp_results.csv')
    sig = pd.read_csv('results/ihdp/ihdp_wilcoxon_ate_error.csv').set_index('method')['reject']

    agg = df.groupby('method').agg(ate=('ate_error', 'mean'), sem=('ate_error', 'sem'),
                                   f1=('f1', 'mean')).sort_values('ate')

    def colour(m):
        if m == 'QMI-CFS':
            return C_QMI
        return C_DIFF if bool(sig.get(m, False)) else C_TIED

    colours = [colour(m) for m in agg.index]
    # In the scatter, significance is also shown by marker shape, not by colour alone.
    is_diff = [m != 'QMI-CFS' and bool(sig.get(m, False)) for m in agg.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: ATE error by method, sorted, bars start at zero.
    x = np.arange(len(agg))
    ax1.bar(x, agg['ate'], yerr=agg['sem'], color=colours, capsize=3,
            edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(agg.index, rotation=35, ha='right', fontsize=9)
    ax1.set_ylabel('ATE error (mean $\\pm$ s.e.)', fontsize=10)
    ax1.set_ylim(0, None)
    ax1.set_title('IHDP-inspired: ATE error by method', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', visible=False)

    # Right: F1 vs ATE error with direct labels.
    markers = ['D' if m == 'QMI-CFS' else ('s' if d else 'o') for m, d in zip(agg.index, is_diff)]
    for (m, row), c, mk in zip(agg.iterrows(), colours, markers):
        ax2.scatter(row['f1'], row['ate'], c=[c], marker=mk, s=70, edgecolors='black',
                    linewidths=0.5, zorder=3)
    offsets = {
        'QMI-CFS': (-6, -13, 'right'), 'LASSO': (6, 6, 'left'),
        'CMI': (0, 10, 'center'), 'HSIC-Lasso': (0, -14, 'center'),
        'RF': (-8, 8, 'right'), 'All-Features': (6, 6, 'left'),
        'HSIC-Med': (7, 0, 'left'), 'HSIC-CV': (7, 0, 'left'),
        'MI': (7, 0, 'left'), 'Boruta': (7, -2, 'left'),
    }
    for m, row in agg.iterrows():
        dx, dy, ha = offsets.get(m, (6, 4, 'left'))
        ax2.annotate(m, (row['f1'], row['ate']), textcoords='offset points',
                     xytext=(dx, dy), ha=ha, va='center', fontsize=8,
                     fontweight='bold' if m == 'QMI-CFS' else 'normal')
    ax2.set_xlabel('F1 score (confounder recovery)', fontsize=10)
    ax2.set_ylabel('ATE error', fontsize=10)
    ax2.set_xlim(0.1, 1.1)
    ax2.set_ylim(0.0, 0.27)
    ax2.set_title('IHDP-inspired: F1 vs. ATE error', fontsize=12, fontweight='bold')

    handles = [mpatches.Patch(facecolor=C_QMI, edgecolor='black', label='QMI-CFS'),
               mpatches.Patch(facecolor=C_TIED, edgecolor='black',
                              label='Not significantly different from QMI-CFS'),
               mpatches.Patch(facecolor=C_DIFF, edgecolor='black',
                              label='Significantly different from QMI-CFS')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _save(fig, 'v6_ihdp_combined')


def plot_f1_vs_ate_by_dataset():
    methods = ['QMI-CFS', 'Boruta', 'CMI', 'LASSO', 'MI']
    df1 = pd.read_csv('results/experiment1/experiment1_results.csv')
    df2 = pd.read_csv('results/experiment2/experiment2_results.csv')
    f1 = df1[df1['method'].isin(methods)].groupby(['dataset', 'method'])['f1'].mean()
    ate = df2[df2['method'].isin(methods)].groupby(['dataset', 'method'])['ate_error'].mean()
    data = pd.concat([f1, ate], axis=1).dropna().reset_index()

    # Distinct marker shapes as well as colours, so methods remain identifiable
    # even where colours are similar (e.g. CMI vs LASSO) or printed in greyscale.
    markers = {'QMI-CFS': '*', 'Boruta': 's', 'CMI': '^', 'LASSO': 'D', 'MI': 'o'}
    sizes = {'QMI-CFS': 260, 'Boruta': 80, 'CMI': 95, 'LASSO': 70, 'MI': 85}

    fig, axes = plt.subplots(3, 2, figsize=(10, 10.5))
    for ax, ds in zip(axes.flatten(), ['LLD', 'NLC', 'HDS', 'CC', 'WC', 'IC']):
        sub = data[data['dataset'] == ds]
        for m in methods:
            row = sub[sub['method'] == m]
            ax.scatter(row['f1'], row['ate_error'], color=METHOD_COLORS[m], marker=markers[m],
                       s=sizes[m], edgecolors='black', linewidths=0.6,
                       zorder=4 if m == 'QMI-CFS' else 3)
        r, p = stats.pearsonr(sub['f1'], sub['ate_error'])
        # Place the statistics box where it does not cover a data point.
        bx, by, ha, va = (0.03, 0.05, 'left', 'bottom') if ds == 'WC' else (0.97, 0.95, 'right', 'top')
        ax.text(bx, by, f'$r={r:.2f}$, $p={p:.2f}$ ($n=5$)', transform=ax.transAxes,
                ha=ha, va=va, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        ax.set_title(ds, fontsize=12, fontweight='bold')
        ax.set_xlabel('F1 score', fontsize=9)
        ax.set_ylabel('ATE error', fontsize=9)

    handles = [plt.Line2D([], [], ls='', marker=markers[m], color=METHOD_COLORS[m],
                          markeredgecolor='black', markeredgewidth=0.6,
                          markersize=15 if m == 'QMI-CFS' else 9, label=m) for m in methods]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.035, 1, 1])
    _save(fig, 'v6_f1_vs_ate_by_dataset')


def _within_draw_rho(df, group_cols):
    """Spearman correlation of F1 vs ATE error across methods inside each draw."""
    rhos = []
    for _, g in df.dropna(subset=['f1', 'ate_error']).groupby(group_cols):
        if g['f1'].nunique() > 1 and g['ate_error'].nunique() > 1:
            rhos.append(stats.spearmanr(g['f1'], g['ate_error'])[0])
    rhos = np.array(rhos)
    half = 1.96 * rhos.std(ddof=1) / np.sqrt(len(rhos))
    return rhos.mean(), half, stats.wilcoxon(rhos)[1]


def plot_f1_ate_summary():
    """Two-panel summary of the recovery-estimation relationship.

    Left:  method means on the six synthetic DGPs, z-scored within each DGP so that
           DGPs with different error scales can be pooled (30 points).
    Right: within-draw Spearman correlation between F1 and ATE error by sample size
           (small-sample experiment) and on the IHDP-inspired benchmark.
    """
    methods = ['QMI-CFS', 'Boruta', 'CMI', 'LASSO', 'MI']
    df1 = pd.read_csv('results/experiment1/experiment1_results.csv')
    df2 = pd.read_csv('results/experiment2/experiment2_results.csv')
    f1 = df1[df1['method'].isin(methods)].groupby(['dataset', 'method'])['f1'].mean()
    ate = df2[df2['method'].isin(methods)].groupby(['dataset', 'method'])['ate_error'].mean()
    data = pd.concat([f1, ate], axis=1).dropna().reset_index()
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    data['f1_z'] = data.groupby('dataset')['f1'].transform(z)
    data['ate_z'] = data.groupby('dataset')['ate_error'].transform(z)
    r, p = stats.pearsonr(data['f1_z'], data['ate_z'])
    rho, p_rho = stats.spearmanr(data['f1_z'], data['ate_z'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) All 30 method x DGP means in one plot, z-scored within each DGP.
    # Each method has its own marker shape and colour (QMI-CFS: large star).
    colours = {'QMI-CFS': '#6A1B9A', 'Boruta': '#0072B2', 'CMI': '#E69F00',
               'LASSO': '#009E73', 'MI': '#8C8C8C'}
    markers = {'QMI-CFS': '*', 'Boruta': 's', 'CMI': '^', 'LASSO': 'D', 'MI': 'o'}
    msizes = {'QMI-CFS': 230, 'Boruta': 60, 'CMI': 75, 'LASSO': 55, 'MI': 65}
    for m in methods:
        sub = data[data['method'] == m]
        ax1.scatter(sub['f1_z'], sub['ate_z'], color=colours[m], marker=markers[m],
                    s=msizes[m], edgecolors='black', linewidths=0.6, label=m,
                    zorder=4 if m == 'QMI-CFS' else 3)
    ax1.axhline(0, color='gray', lw=0.8)
    ax1.axvline(0, color='gray', lw=0.8)
    ax1.text(0.97, 0.96, f'Pearson $r={r:.2f}$ ($p={p:.2f}$)\nSpearman $\\rho={rho:.2f}$\n30 method$\\times$DGP means',
             transform=ax1.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    ax1.set_xlabel('F1 (z-score within DGP; right = better recovery)', fontsize=9)
    ax1.set_ylabel('ATE error (z-score within DGP; down = better)', fontsize=9)
    ax1.set_title('(a) Six synthetic DGPs, method means', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=5,
               frameon=False, markerscale=0.9)

    exp3 = pd.read_csv('results/experiment3/experiment3_results.csv')
    sizes = sorted(exp3['n_samples'].unique())
    stats_n = [_within_draw_rho(exp3[exp3['n_samples'] == n], 'replication') for n in sizes]
    ihdp = pd.read_csv('results/ihdp/ihdp_results.csv')
    m_i, h_i, p_i = _within_draw_rho(ihdp, 'seed')

    xs = np.arange(len(sizes))
    means = [s[0] for s in stats_n]
    halves = [s[1] for s in stats_n]
    ax2.axhline(0, color='gray', lw=1, ls='--', zorder=1)
    ax2.errorbar(xs, means, yerr=halves, fmt='o-', color=C_QMI, capsize=4, lw=1.5,
                 ms=7, zorder=3, label='Small-sample DGP')
    ax2.errorbar([len(sizes) + 0.4], [m_i], yerr=[h_i], fmt='s', color=C_TIED, capsize=4,
                 ms=7, zorder=3, label='IHDP-inspired ($n=747$)')
    ax2.set_xticks(list(xs) + [len(sizes) + 0.4])
    ax2.set_xticklabels([f'$n={n}$' for n in sizes] + ['IHDP-insp.'], fontsize=9)
    ax2.set_xlim(-0.5, len(sizes) + 0.9)
    ax2.set_ylabel('Within-draw Spearman $\\rho$\n(F1 vs. ATE error, mean $\\pm$ 95% CI)', fontsize=10)
    ax2.set_ylim(-0.5, 0.3)
    ax2.set_title('(b) Identical draws: does higher F1 mean lower error?', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    ax2.text(0.02, 0.97, 'below 0: higher F1 goes with lower ATE error', transform=ax2.transAxes,
             fontsize=8, va='top', color='dimgray')

    plt.tight_layout()
    _save(fig, 'v7_f1_ate_summary')
    print(f'Left panel: pooled within-DGP Pearson r={r:.3f} (p={p:.3f}), Spearman rho={rho:.3f} (p={p_rho:.3f})')


def plot_same_draw_rho():
    """Single-panel figure: within-draw F1-vs-ATE correlation by sample size."""
    exp3 = pd.read_csv('results/experiment3/experiment3_results.csv')
    sizes = sorted(exp3['n_samples'].unique())
    res = [_within_draw_rho(exp3[exp3['n_samples'] == n], 'replication') for n in sizes]
    m_i, h_i, _ = _within_draw_rho(pd.read_csv('results/ihdp/ihdp_results.csv'), 'seed')

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = np.arange(len(sizes))
    x_ihdp = len(sizes) + 0.4
    ax.axhspan(-0.5, 0, color='#2a9d8f', alpha=0.07, zorder=0)
    ax.axhline(0, color='gray', lw=1, ls='--', zorder=1)
    ax.errorbar(xs, [r[0] for r in res], yerr=[r[1] for r in res], fmt='o-', color=C_QMI,
                capsize=4, lw=1.8, ms=7, zorder=3, label='Small-sample DGP')
    ax.errorbar([x_ihdp], [m_i], yerr=[h_i], fmt='s', color=C_TIED, capsize=4, ms=7,
                zorder=3, label='IHDP-inspired ($n=747$)')
    ax.set_xticks(list(xs) + [x_ihdp])
    ax.set_xticklabels([f'$n={n}$' for n in sizes] + ['IHDP-\ninspired'], fontsize=9)
    ax.set_xlim(-0.5, x_ihdp + 0.5)
    ax.set_ylim(-0.45, 0.25)
    ax.set_ylabel('Correlation of F1 with ATE error\n(within draw, mean $\\pm$ 95% CI)', fontsize=9)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, 'v8_same_draw_rho')


if __name__ == '__main__':
    # Only the figures used in the final submission; the other plot functions
    # produce earlier drafts kept in paper/archive/figures.
    plot_ihdp_combined()
    plot_same_draw_rho()
