#!/usr/bin/env python3
"""Run all three experiments sequentially."""
import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='Run all QMI-CFS experiments')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--skip-exp1', action='store_true')
    parser.add_argument('--skip-exp2', action='store_true')
    parser.add_argument('--skip-exp3', action='store_true')
    args = parser.parse_args()

    experiments = [
        ('Experiment 1: Feature Selection', 'run_experiment1.py', args.skip_exp1),
        ('Experiment 2: Treatment Effect Estimation', 'run_experiment2.py', args.skip_exp2),
        ('Experiment 3: Small-Sample Analysis', 'run_experiment3.py', args.skip_exp3),
    ]

    for name, script, skip in experiments:
        if skip:
            print(f"\n{'='*60}")
            print(f"SKIPPING: {name}")
            continue
        print(f"\n{'='*60}")
        print(f"RUNNING: {name}")
        print('='*60)
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, script),
               '--output', os.path.join(args.output, script.replace('run_', '').replace('.py', ''))]
        subprocess.run(cmd, check=False)

    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results directory: {args.output}")


if __name__ == '__main__':
    main()
