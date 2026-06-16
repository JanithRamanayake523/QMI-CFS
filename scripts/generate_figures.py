#!/usr/bin/env python3
"""Generate paper manuscript figures from existing experimental results."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
from qmi_cfs.plot_style import set_publication_style, METHOD_COLORS, get_color
set_publication_style()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from qmi_cfs.utils import ensure_dir


FINAL_METHODS = ['QMI-CFS', 'Boruta', 'CMI', 'LASSO', 'MI', 'All-Features']


def _filter_methods(df):
    """Keep only methods that are part of the final paper comparison."""
    return df[df['method'].isin(FINAL_METHODS)].copy()


def load_aggregated_data():
    """Aggregate F1 and ATE per (dataset, method)."""
    df1 = pd.read_csv('results/experiment1/experiment1_results.csv')
    df2 = pd.read_csv('results/experiment2/experiment2_results.csv')
    df_ihdp = pd.read_csv('results/ihdp/ihdp_results.csv')

    df1 = _filter_methods(df1)
    df2 = _filter_methods(df2)
    df_ihdp = _filter_methods(df_ihdp)

    f1_summary = df1.groupby(['dataset', 'method'])['f1'].agg(['mean', 'std']).reset_index()
    f1_summary.columns = ['dataset', 'method', 'f1_mean', 'f1_std']

    ate_summary = df2.groupby(['dataset', 'method'])['ate_error'].agg(['mean', 'std']).reset_index()
    ate_summary.columns = ['dataset', 'method', 'ate_mean', 'ate_std']

    merged = pd.merge(f1_summary, ate_summary, on=['dataset', 'method'], how='inner')

    ihdp_summary = df_ihdp.groupby(['dataset', 'method']).agg({
        'f1': ['mean', 'std'],
        'ate_error': ['mean', 'std']
    }).reset_index()
    ihdp_summary.columns = ['dataset', 'method', 'f1_mean', 'f1_std', 'ate_mean', 'ate_std']

    all_data = pd.concat([merged, ihdp_summary], ignore_index=True)
    return all_data


def plot_f1_vs_ate_by_dataset(data, output_path):
    """Dataset-wise F1 vs ATE scatter panels (3x2 grid, synthetic DGPs only)."""
    method_order = FINAL_METHODS
    dataset_order = ['LLD', 'NLC', 'HDS', 'CC', 'WC', 'IC']

    palette = METHOD_COLORS

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, ds in enumerate(dataset_order):
        ax = axes[idx]
        sub = data[data['dataset'] == ds]
        if sub.empty:
            ax.set_visible(False)
            continue

        for method in method_order:
            row = sub[sub['method'] == method]
            if row.empty:
                continue
            ax.scatter(
                row['f1_mean'].values[0],
                row['ate_mean'].values[0],
                color=palette.get(method, 'black'),
                s=90,
                alpha=0.85,
                edgecolors='white',
                linewidths=0.5,
                label=method if idx == 0 else "",
                zorder=3,
            )

        # Per-dataset regression line
        if len(sub) >= 2:
            try:
                slope, intercept, r_value, p_value, _ = stats.linregress(
                    sub['f1_mean'], sub['ate_mean']
                )
                x_min, x_max = sub['f1_mean'].min(), sub['f1_mean'].max()
                if x_max > x_min:
                    x_line = np.linspace(x_min, x_max, 100)
                    ax.plot(x_line, slope * x_line + intercept, '--',
                            color='gray', alpha=0.6, linewidth=1.2, zorder=1)
                ax.text(0.95, 0.95,
                        f'$r={r_value:.2f}$\n$p={p_value:.3f}$',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=8, bbox=dict(boxstyle='round', facecolor='white',
                                              edgecolor='gray', alpha=0.8))
            except Exception:
                pass

        ax.set_title(ds, fontsize=12, fontweight='bold')
        ax.set_xlabel('F1 Score', fontsize=9)
        ax.set_ylabel('ATE Error', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(dataset_order), len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    handles = [mpatches.Patch(color=palette.get(m, 'black'), label=m)
               for m in method_order if m in data['method'].values]
    fig.legend(handles, [m for m in method_order if m in data['method'].values],
               loc='lower center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Feature Recovery vs. Causal Estimation Quality by Dataset',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    _save_fig(fig, output_path)


def plot_ihdp_combined(data, output_path):
    """IHDP Setting A: ATE error and F1 vs ATE in one figure."""
    method_order = FINAL_METHODS
    palette = METHOD_COLORS

    df_raw = pd.read_csv('results/ihdp/ihdp_results.csv')
    df_raw = _filter_methods(df_raw)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: ATE error by method (single-panel version of Figure 2 for IHDP).
    agg = df_raw.groupby('method')['ate_error'].agg(['mean', 'sem']).reindex(method_order)
    x = np.arange(len(method_order))
    ax1.bar(x, agg['mean'], yerr=agg['sem'],
            color=[palette.get(m, '#7f8c8d') for m in method_order],
            alpha=0.85, capsize=3, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_order, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('ATE Error', fontsize=10)
    ax1.set_title('IHDP: ATE Error by Method', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Right: F1 vs ATE scatter for IHDP only.
    sub = data[data['dataset'] == 'IHDP-A']
    for method in method_order:
        row = sub[sub['method'] == method]
        if row.empty:
            continue
        ax2.scatter(row['f1_mean'].values[0],
                    row['ate_mean'].values[0],
                    color=palette.get(method, '#7f8c8d'),
                    s=90, alpha=0.85,
                    edgecolors='white', linewidths=0.5,
                    label=method)

    # Regression line for IHDP recovery--estimation relationship.
    if len(sub) >= 2:
        slope, intercept, r_value, p_value, _ = stats.linregress(
            sub['f1_mean'], sub['ate_mean']
        )
        x_min, x_max = sub['f1_mean'].min(), sub['f1_mean'].max()
        if x_max > x_min:
            x_line = np.linspace(x_min, x_max, 100)
            ax2.plot(x_line, slope * x_line + intercept, '--',
                     color='gray', alpha=0.7, linewidth=1.5, zorder=1)
        ax2.text(0.95, 0.95,
                 f'$r={r_value:.2f}$\n$p={p_value:.3f}$',
                 transform=ax2.transAxes, ha='right', va='top',
                 fontsize=8, bbox=dict(boxstyle='round', facecolor='white',
                                       edgecolor='gray', alpha=0.8))

    for method, xytext in [('QMI-CFS', (8, 5)), ('CMI', (-40, 5))]:
        row = sub[sub['method'] == method]
        if not row.empty:
            ax2.annotate(method,
                         (row['f1_mean'].values[0], row['ate_mean'].values[0]),
                         textcoords='offset points', xytext=xytext,
                         fontsize=8,
                         arrowprops=dict(arrowstyle='->', lw=0.7, color='gray'))

    ax2.set_xlabel('F1 Score', fontsize=10)
    ax2.set_ylabel('ATE Error', fontsize=10)
    ax2.set_title('IHDP: F1 vs. ATE Error', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, output_path)


def _save_fig(fig, output_path):
    """Save figure as PDF and PNG for paper and README previews."""
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    png_path = os.path.splitext(output_path)[0] + '.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_path} and {png_path}')


def print_correlations(data):
    """Print correlation statistics for inclusion in paper."""
    r_all, p_all = stats.pearsonr(data['f1_mean'], data['ate_mean'])
    rho_all, _ = stats.spearmanr(data['f1_mean'], data['ate_mean'])
    print(f'All datasets: Pearson r={r_all:.3f}, p={p_all:.3f}, R^2={r_all**2:.3f}; Spearman rho={rho_all:.3f}')

    sub = data[data['dataset'] != 'IC']
    r_no_ic, p_no_ic = stats.pearsonr(sub['f1_mean'], sub['ate_mean'])
    rho_no_ic, _ = stats.spearmanr(sub['f1_mean'], sub['ate_mean'])
    print(f'Excluding IC: Pearson r={r_no_ic:.3f}, p={p_no_ic:.3f}, R^2={r_no_ic**2:.3f}; Spearman rho={rho_no_ic:.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='paper/figures')
    args = parser.parse_args()

    ensure_dir(args.output)
    data = load_aggregated_data()
    print_correlations(data)

    plot_f1_vs_ate_by_dataset(data, os.path.join(args.output, 'v5_f1_vs_ate_by_dataset.pdf'))
    plot_ihdp_combined(data, os.path.join(args.output, 'v5_ihdp_combined.pdf'))


if __name__ == '__main__':
    main()
