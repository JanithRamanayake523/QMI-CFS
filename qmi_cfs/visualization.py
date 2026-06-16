"""Publication-quality visualization for QMI-CFS experiments."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from typing import Optional

from qmi_cfs.plot_style import set_publication_style, get_color

set_publication_style()


METHOD_ORDER = ['QMI-CFS', 'Boruta', 'CMI', 'LASSO', 'MI', 'All-Features']


def plot_experiment1_results(df: pd.DataFrame, output_path: str,
                              figsize=(12, 6)) -> None:
    """Grouped bar chart: mean F1 by method across datasets."""
    agg = df.groupby(['dataset', 'method'])['f1'].agg(['mean', 'sem']).reset_index()
    datasets = agg['dataset'].unique()
    methods = agg['method'].unique()

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        data = agg[agg['method'] == method]
        means = [data[data['dataset'] == d]['mean'].values[0] if len(data[data['dataset'] == d]) > 0 else 0
                 for d in datasets]
        sems = [data[data['dataset'] == d]['sem'].values[0] if len(data[data['dataset'] == d]) > 0 else 0
                for d in datasets]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=sems, label=method,
               color=get_color(method), alpha=0.85, capsize=2)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Experiment 1: Feature Selection Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_experiment2_results(df: pd.DataFrame, output_path: str,
                              figsize=(12, 10)) -> None:
    """3x2 grid: mean ATE error by method, one panel per dataset.

    Each panel has its own y-axis so differences between methods are visible
    even when datasets have very different ATE-error scales.
    """
    method_order = ['QMI-CFS', 'Boruta', 'CMI', 'LASSO', 'MI', 'All-Features']
    dataset_order = ['LLD', 'NLC', 'HDS', 'CC', 'WC', 'IC']

    agg = (
        df[df['method'].isin(method_order)]
        .groupby(['dataset', 'method'])['ate_error']
        .agg(['mean', 'sem'])
        .reset_index()
    )

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharey=False)
    axes = axes.flatten()

    width = 0.6
    for idx, ds in enumerate(dataset_order):
        ax = axes[idx]
        sub = agg[agg['dataset'] == ds]
        if sub.empty:
            ax.set_visible(False)
            continue

        # Order methods consistently
        sub = sub.set_index('method').reindex(method_order).reset_index()
        means = sub['mean'].fillna(0).values
        sems = sub['sem'].fillna(0).values
        x = np.arange(len(method_order))

        bars = ax.bar(x, means, width, yerr=sems,
                      color=[get_color(m) for m in method_order],
                      alpha=0.85, capsize=3, edgecolor='black', linewidth=0.5)

        ax.set_title(ds, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_order, fontsize=9, rotation=30, ha='right')
        ax.set_ylabel('ATE Error', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Per-panel y-axis focused on the observed range
        upper = np.max(means + sems) if np.any(means + sems > 0) else 1.0
        lower = np.min(np.maximum(means - sems, 0))
        span = upper - lower
        if span <= 1e-9:
            span = 0.1 * upper if upper > 0 else 1.0
        pad = 0.15 * span
        ax.set_ylim(max(0, lower - pad), upper + pad)

    # Shared legend on the empty right side or below
    handles = [mpatches.Patch(color=get_color(m), label=m) for m in method_order]
    fig.legend(handles, method_order, loc='lower center', ncol=len(method_order),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('ATE Estimation Error by Dataset and Method', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_experiment3_results(df: pd.DataFrame, output_path: str,
                              figsize=(14, 10)) -> None:
    """Two-panel figure: ATE Error and F1 vs. Sample Size."""
    method_order = ['QMI-CFS', 'Boruta', 'CMI', 'LASSO', 'MI', 'All-Features']
    df = df[df['method'].isin(method_order)].copy()
    agg_ate = df.groupby(['n_samples', 'method'])['ate_error'].agg(['mean', 'sem']).reset_index()
    agg_f1 = df.groupby(['n_samples', 'method'])['f1'].agg(['mean', 'sem']).reset_index()
    methods = [m for m in method_order if m in df['method'].values]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Top: ATE Error
    for method in methods:
        data = agg_ate[agg_ate['method'] == method]
        ax1.errorbar(data['n_samples'], data['mean'], yerr=data['sem'],
                     marker='o', label=method, color=get_color(method),
                     capsize=3, linewidth=2, markersize=6)
    ax1.set_xscale('log')
    ax1.set_xlabel('Sample Size', fontsize=12)
    ax1.set_ylabel('ATE Error', fontsize=12)
    ax1.set_title('ATE Error vs. Sample Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: F1 Score
    for method in methods:
        data = agg_f1[agg_f1['method'] == method]
        ax2.errorbar(data['n_samples'], data['mean'], yerr=data['sem'],
                     marker='s', label=method, color=get_color(method),
                     capsize=3, linewidth=2, markersize=6)
    ax2.set_xscale('log')
    ax2.set_xlabel('Sample Size', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score vs. Sample Size', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_table(df: pd.DataFrame, group_cols: list,
                         metric_cols: list) -> pd.DataFrame:
    """Create publication-ready summary table with mean ± std format."""
    agg = df.groupby(group_cols)[metric_cols].agg(['mean', 'std'])
    formatted = pd.DataFrame(index=agg.index)
    for col in metric_cols:
        formatted[col] = (agg[(col, 'mean')].round(4).astype(str) +
                          ' ± ' + agg[(col, 'std')].round(4).astype(str))
    return formatted
