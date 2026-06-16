#!/usr/bin/env python3
"""Generate publication-ready LaTeX tables for the QMI-CFS paper.

Usage:
    python scripts/generate_tables.py --output paper/tables
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from qmi_cfs.utils import ensure_dir


def row_end():
    """Return LaTeX row ending."""
    return " " + "\\\\" + "[2pt]\n"

def get_mean_std(summary, dataset, method, metric='mean'):
    """Extract mean/std for a dataset/method from summary DataFrame."""
    sub = summary[(summary['dataset'] == dataset) & (summary['method'] == method)]
    if sub.empty:
        return None, None
    return sub[metric].values[0], sub['std'].values[0]


def generate_table1_datasets():
    """Table 1: Dataset characteristics."""
    rows = [
        ("LLD (Linear Low-Dim)", 500, 20, 4, 4, 4, 2.0),
        ("NLC (Nonlinear Conf.)", 500, 20, 4, 4, 4, 1.5),
        ("HDS (High-Dim Sparse)", 500, 100, 5, 5, 5, 1.0),
        ("CC (Correlated Conf.)", 500, 20, 6, 4, 4, 2.0),
        ("WC (Weak Confounding)", 500, 15, 3, 3, 3, 1.0),
        ("IC (Interacting Conf.)", 500, 20, 6, 0, 0, 2.0),
        ("IHDP (Setting A)", 747, 25, 3, "-", "-", 4.0),
    ]
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Dataset characteristics. $n_C$: confounders, $n_I$: instruments, $n_P$: precision variables.}
\\label{tab:datasets}
\\begin{tabular}{lcccccc}
\\hline
\\rule{0pt}{12pt}Dataset & $n$ & $d$ & $n_C$ & $n_I$ & $n_P$ & True ATE \\\\
"""
    for r in rows:
        latex += f"\\rule{{0pt}}{{12pt}}{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]:.1f} \\\\\n"
    latex += r"""\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_table2_f1(df_exp1, ttest_df):
    """Table 2: Feature selection F1 scores."""
    df = df_exp1[df_exp1['method'] != 'QMI-CFS-Classic'].copy()
    summary = df.groupby(['dataset', 'method'])['f1'].agg(['mean', 'std']).reset_index()
    summary.columns = ['dataset', 'method', 'mean', 'std']

    method_order = ['QMI-CFS', 'Boruta', 'CMI', 'LASSO', 'MI']
    datasets = ['LLD', 'NLC', 'HDS', 'CC', 'WC', 'IC']

    sig_map = {}
    if ttest_df is not None:
        for _, row in ttest_df.iterrows():
            if row['baseline'] == 'QMI-CFS':
                sig_map[(row['dataset'], row['method'])] = row.get('significance', '')

    cols = 'l' + 'c' * len(method_order)
    header = ' & '.join(['Dataset'] + method_order)
    latex = f"""\\begin{{table*}}[tbp]
\\centering
\\caption{{Feature selection accuracy (F1 score, mean $\\pm$ std). Best mean F1 in bold. Significance markers for QMI-CFS vs. baseline: $^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$. Gray markers indicate baselines significantly better than QMI-CFS.}}
\\label{{tab:f1}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{2pt}}
\\begin{{tabular}}{{{cols}}}
\\hline
\\rule{{0pt}}{{12pt}}{header} \\\\
"""
    for ds in datasets:
        row = f"\\rule{{0pt}}{{12pt}}{ds}"
        means = {m: summary[(summary['dataset']==ds)&(summary['method']==m)]['mean'].values[0]
                 for m in method_order if not summary[(summary['dataset']==ds)&(summary['method']==m)].empty}
        best_m = max(means.values()) if means else 0
        for m in method_order:
            if m not in means:
                row += " & --"
                continue
            mean = summary[(summary['dataset']==ds)&(summary['method']==m)]['mean'].values[0]
            std = summary[(summary['dataset']==ds)&(summary['method']==m)]['std'].values[0]
            val_str = f"{mean:.3f} \\pm {std:.3f}"
            if abs(mean - best_m) < 1e-9:
                cell = f"$\\mathbf{{{val_str}}}$"
            else:
                cell = f"${val_str}$"
            sig = sig_map.get((ds, m), '')
            if sig:
                sub = ttest_df[(ttest_df['dataset'] == ds) & (ttest_df['method'] == m)]
                if not sub.empty:
                    mean_diff = sub['mean_diff'].values[0]
                    if mean_diff < 0:  # baseline better
                        inner = cell.replace('$', '')
                        cell = f"$\\textcolor{{gray}}{{{inner}}}^{{{sig}}}$"
                    else:
                        cell = cell[:-1] + f"^{{{sig}}}$"
            row += f" & {cell}"
        row += row_end()
        latex += row
    latex += """\\hline
\\end{tabular}
\\end{table*}
"""
    return latex


def generate_table3_ate(df_exp2):
    """Table 3: ATE estimation error."""
    summary = df_exp2.groupby(['dataset', 'method'])['ate_error'].agg(['mean', 'std']).reset_index()
    summary.columns = ['dataset', 'method', 'mean', 'std']

    method_order = ['QMI-CFS', 'MI', 'LASSO', 'Boruta', 'CMI',
                    'All-Features']
    datasets = ['LLD', 'NLC', 'HDS', 'CC', 'WC', 'IC']

    cols = 'l' + 'c' * len(method_order)
    header = ' & '.join(['Dataset'] + method_order)
    latex = f"""\\begin{{table*}}[tbp]
\\centering
\\caption{{ATE estimation error (mean $\\pm$ std). Lowest mean error in bold.}}
\\label{{tab:ate}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{2pt}}
\\begin{{tabular}}{{{cols}}}
\\hline
\\rule{{0pt}}{{12pt}}{header} \\\\
"""
    for ds in datasets:
        row = f"\\rule{{0pt}}{{12pt}}{ds}"
        means = {m: summary[(summary['dataset']==ds)&(summary['method']==m)]['mean'].values[0]
                 for m in method_order if not summary[(summary['dataset']==ds)&(summary['method']==m)].empty}
        best_m = min(means.values()) if means else 0
        for m in method_order:
            if m not in means:
                row += " & --"
                continue
            mean = summary[(summary['dataset']==ds)&(summary['method']==m)]['mean'].values[0]
            std = summary[(summary['dataset']==ds)&(summary['method']==m)]['std'].values[0]
            val_str = f"{mean:.3f} \\pm {std:.3f}"
            if abs(mean - best_m) < 1e-9:
                cell = f"$\\mathbf{{{val_str}}}$"
            else:
                cell = f"${val_str}$"
            row += f" & {cell}"
        row += row_end()
        latex += row
    latex += """\\hline
\\end{tabular}
\\end{table*}
"""
    return latex


def generate_table4_small_sample(df_exp3):
    """Table 4: Small-sample F1 and ATE error."""
    summary = df_exp3.groupby(['n_samples', 'method']).agg({
        'f1': ['mean', 'std'],
        'ate_error': ['mean', 'std']
    }).reset_index()
    # Flatten multi-index columns
    summary.columns = [' '.join(col).strip() if col[1] not in ['nan', ''] else col[0]
                       for col in summary.columns.values]
    # Rename to simpler names
    summary = summary.rename(columns={
        'n_samples ': 'n_samples',
        'method ': 'method',
        'f1 mean': 'f1_mean',
        'f1 std': 'f1_std',
        'ate_error mean': 'ate_mean',
        'ate_error std': 'ate_std',
    })
    method_order = ['QMI-CFS', 'Boruta', 'CMI', 'MI', 'LASSO',
                    'All-Features']
    sample_sizes = sorted(df_exp3['n_samples'].unique())

    cols = 'l' + 'c' * len(method_order)
    header = ' & '.join(['$n$'] + method_order)
    latex = f"""\\begin{{table*}}[tbp]
\\centering
\\caption{{Small-sample analysis: F1 score and ATE error (mean $\\pm$ std).}}
\\label{{tab:small_sample}}
\\footnotesize
\\begin{{tabular}}{{{cols}}}
\\hline
\\rule{{0pt}}{{12pt}}{header} \\\\
"""
    for metric, label, mean_col, std_col in [
        ('f1', 'F1', 'f1_mean', 'f1_std'),
        ('ate_error', 'ATE Error', 'ate_mean', 'ate_std')
    ]:
        latex += f"\\multicolumn{{7}}{{l}}{{\\textit{{{label}}}}} \\\\\n"
        for n in sample_sizes:
            row = f"\\rule{{0pt}}{{12pt}}{n}"
            for m in method_order:
                sub = summary[(summary['n_samples'] == n) & (summary['method'] == m)]
                if sub.empty:
                    row += " & --"
                else:
                    mean = sub[mean_col].values[0]
                    std = sub[std_col].values[0]
                    row += f" & ${mean:.3f} \\pm {std:.3f}$"
            row += row_end()
            latex += row
        latex += "\\hline\n"
    latex += """\\end{tabular}
\\end{table*}
"""
    return latex


def generate_table5_ihdp(df_ihdp):
    """Table 5: IHDP results."""
    summary = df_ihdp.groupby('method').agg({
        'f1': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'ate_error': ['mean', 'std'],
    }).reset_index()
    summary.columns = ['method', 'f1_mean', 'f1_std', 'p_mean', 'p_std',
                       'r_mean', 'r_std', 'a_mean', 'a_std']
    method_order = ['QMI-CFS', 'Boruta', 'CMI', 'LASSO', 'MI',
                    'All-Features']

    latex = r"""\begin{table}[htbp]
\centering
\caption{IHDP Setting A results ($n=747$, true ATE = 4.0). Best mean ATE error and best mean F1 in bold.}
\label{tab:ihdp}
\small
\begin{tabular}{lcccc}
\hline
\rule{0pt}{12pt}Method & \multicolumn{1}{l}{F1} & \multicolumn{1}{l}{ATE Error} & Precision & Recall \\\\
"""
    best_f1 = summary['f1_mean'].max()
    best_ate = summary['a_mean'].min()
    for m in method_order:
        sub = summary[summary['method'] == m]
        if sub.empty:
            continue
        f1_m = sub['f1_mean'].values[0]
        f1_s = sub['f1_std'].values[0]
        p_m = sub['p_mean'].values[0]
        p_s = sub['p_std'].values[0]
        r_m = sub['r_mean'].values[0]
        r_s = sub['r_std'].values[0]
        a_m = sub['a_mean'].values[0]
        a_s = sub['a_std'].values[0]

        f1_str = f"${f1_m:.3f} \\pm {f1_s:.3f}$"
        ate_str = f"${a_m:.3f} \\pm {a_s:.3f}$"
        if abs(f1_m - best_f1) < 1e-9:
            f1_str = f"$\\mathbf{{{f1_m:.3f} \\pm {f1_s:.3f}}}$"
        if abs(a_m - best_ate) < 1e-9:
            ate_str = f"$\\mathbf{{{a_m:.3f} \\pm {a_s:.3f}}}$"

        latex += f"\\rule{{0pt}}{{12pt}}{m} & {f1_str} & {ate_str} & ${p_m:.3f} \\pm {p_s:.3f}$ & ${r_m:.3f} \\pm {r_s:.3f}$ \\\\\n"
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    return latex


def generate_table6_ablation(df_abl):
    """Table 6: Ablation study."""
    summary = df_abl.groupby(['dataset', 'method'])['f1'].agg(['mean', 'std']).reset_index()
    summary.columns = ['dataset', 'method', 'mean', 'std']

    variant_order = ['QMI-CFS (Full)', 'QMI-CFS (No Bootstrap)', 'QMI-CFS (No Pairwise)',
                     'QMI-CFS (Conservative)', 'QMI-CFS (Precision)']
    datasets = ['LLD', 'NLC', 'HDS', 'CC', 'WC', 'IC']

    latex = """\\begin{table}[htbp]
\\centering
\\caption{Ablation study: Mean F1 score (mean $\\pm$ std) across QMI-CFS variants.}
\\label{tab:ablation}
\\small
\\begin{tabular}{lccccc}
\\hline
\\rule{0pt}{12pt}Dataset & Full & No Bootstrap & No Pairwise & Conservative & Precision \\\\
"""
    for ds in datasets:
        row = f"\\rule{{0pt}}{{12pt}}{ds}"
        for v in variant_order:
            sub = summary[(summary['dataset']==ds)&(summary['method']==v)]
            if sub.empty:
                row += " & --"
            else:
                mean = sub['mean'].values[0]
                std = sub['std'].values[0]
                row += f" & ${mean:.3f} \\pm {std:.3f}$"
        row += row_end()
        latex += row
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    return latex


def generate_table7_ic_ate(df_exp2, df_exp2_nl):
    """Table 7: IC focused results and nonlinear AIPW."""
    lin = df_exp2[(df_exp2['dataset'] == 'IC') & (df_exp2['method'] == 'QMI-CFS')]['ate_error']
    nln = df_exp2_nl[(df_exp2_nl['dataset'] == 'IC') & (df_exp2_nl['method'] == 'QMI-CFS')]['ate_error']

    latex = """\\begin{table}[htbp]
\\centering
\\caption{Interacting Confounders (IC) dataset: ATE error for QMI-CFS with linear vs. nonlinear AIPW, and baseline methods with linear AIPW.}
\\label{tab:ic_ate}
\\begin{tabular}{lc}
\\hline
\\rule{0pt}{12pt}Method & ATE Error \\\\
"""
    for m in ['MI', 'CMI', 'LASSO', 'Boruta', 'All-Features']:
        sub = df_exp2[(df_exp2['dataset'] == 'IC') & (df_exp2['method'] == m)]['ate_error']
        if len(sub) > 0:
            latex += f"\\rule{{0pt}}{{12pt}}{m} & ${sub.mean():.3f} \\pm {sub.std():.3f}$ \\\\\n"
    latex += f"\\rule{{0pt}}{{12pt}}QMI-CFS (linear AIPW) & ${lin.mean():.3f} \\pm {lin.std():.3f}$ \\\\\n"
    latex += f"\\rule{{0pt}}{{12pt}}QMI-CFS (nonlinear AIPW) & ${nln.mean():.3f} \\pm {nln.std():.3f}$ \\\\\n"
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description='Generate V3 publication tables')
    parser.add_argument('--output', type=str, default='results/generated_tables_v4')
    args = parser.parse_args()

    ensure_dir(args.output)

    df_exp1 = pd.read_csv('results/experiment1/experiment1_results.csv')
    ttest_df = pd.read_csv('results/experiment1/experiment1_ttest.csv')
    df_exp2 = pd.read_csv('results/experiment2/experiment2_results.csv')
    df_exp2_nl = pd.read_csv('results/exp2_nonlinear/experiment2_results.csv')
    df_exp3 = pd.read_csv('results/experiment3/experiment3_results.csv')
    df_ihdp = pd.read_csv('results/ihdp/ihdp_results.csv')
    df_abl = pd.read_csv('results/ablation/ablation_results.csv')

    tables = {
        'table1_datasets.txt': generate_table1_datasets(),
        'table2_f1.txt': generate_table2_f1(df_exp1, ttest_df),
        'table3_ate.txt': generate_table3_ate(df_exp2),
        'table4_small_sample.txt': generate_table4_small_sample(df_exp3),
        'table5_ihdp.txt': generate_table5_ihdp(df_ihdp),
        'table6_ablation.txt': generate_table6_ablation(df_abl),
        'table7_ic_ate.txt': generate_table7_ic_ate(df_exp2, df_exp2_nl),
    }

    for fname, latex in tables.items():
        path = os.path.join(args.output, fname)
        with open(path, 'w') as f:
            f.write(latex)
        print(f"Saved {path}")

    print("\nAll tables generated.")


if __name__ == '__main__':
    main()
