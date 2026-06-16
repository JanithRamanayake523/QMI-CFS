#!/usr/bin/env python3
"""Regenerate paper manuscript figures from existing CSVs using the updated style."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from qmi_cfs.visualization import plot_experiment2_results, plot_experiment3_results
from scripts.generate_ic_figure import plot_ic_comparison

OUTPUT_DIR = 'paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_exp1 = pd.read_csv('results/experiment1/experiment1_results.csv')
df_exp2 = pd.read_csv('results/experiment2/experiment2_results.csv')
df_exp3 = pd.read_csv('results/experiment3/experiment3_results.csv')

plot_experiment2_results(df_exp2, os.path.join(OUTPUT_DIR, 'v3_experiment2_ate.pdf'))
plot_experiment3_results(df_exp3, os.path.join(OUTPUT_DIR, 'v3_small_sample.pdf'))
plot_ic_comparison(df_exp1, os.path.join(OUTPUT_DIR, 'v3_ic_comparison.pdf'))

print('Done regenerating ATE grid, small sample, and IC comparison figures.')
