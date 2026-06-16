#!/usr/bin/env python3
"""Generate the Interacting Confounders (IC) comparison figure for the paper."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from qmi_cfs.plot_style import set_publication_style, get_color
set_publication_style()
import matplotlib.pyplot as plt

from qmi_cfs.utils import ensure_dir


def plot_ic_comparison(df_exp1, output_path):
    """Bar chart comparing methods on the IC dataset."""
    df = df_exp1[df_exp1['dataset'] == 'IC'].copy()
    agg = df.groupby('method')['f1'].agg(['mean', 'std']).reset_index()
    methods = ['QMI-CFS', 'QMI-CFS-Classic', 'MI', 'CMI', 'LASSO', 'Boruta']
    agg = agg.set_index('method').reindex(methods).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods))
    colors = [get_color(m) for m in methods]
    ax.bar(x, agg['mean'], yerr=agg['std'], color=colors, alpha=0.85, capsize=4,
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Interacting Confounders (IC): Feature Selection F1',
                 fontsize=13, fontweight='bold')
    ax.set_ylim([0, 0.7])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate IC comparison figure for the paper manuscript'
    )
    parser.add_argument('--output', type=str, default='paper/figures')
    args = parser.parse_args()

    ensure_dir(args.output)
    df_exp1 = pd.read_csv('results/experiment1/experiment1_results.csv')
    plot_ic_comparison(df_exp1, os.path.join(args.output, 'v3_ic_comparison.pdf'))


if __name__ == '__main__':
    main()
