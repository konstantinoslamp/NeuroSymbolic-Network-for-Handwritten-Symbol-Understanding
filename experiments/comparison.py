"""
comparison.py
Load ablation results and print a formatted comparison table.
Can also run all three strategies inline if no results file exists yet.

Usage:
    python experiments/comparison.py                          # run everything
    python experiments/comparison.py --results results/ablations.json  # just print table
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def print_table(results: dict):
    """Print a human-readable comparison table from ablation results dict."""
    print("\n" + "=" * 80)
    print("  RESULTS COMPARISON: Tsamoura et al. Replication")
    print("=" * 80)

    configs = list(results.keys())

    # Header
    col = 28
    header = f"{'Metric':<35}"
    for name in configs:
        label = results[name].get('config', name)
        header += f"  {label[:col]:<{col}}"
    print(header)
    print("-" * 80)

    def _get(r, *keys, default='N/A'):
        try:
            v = r
            for k in keys:
                v = v[k]
            return f"{float(v):.4f}"
        except (KeyError, TypeError, ValueError):
            return default

    rows = [
        ("Final Train Loss",      lambda r: _get(r, 'training_history', -1, 'avg_loss')),
        ("Final Train Accuracy",  lambda r: _get(r, 'training_history', -1, 'accuracy')),
        ("Train Time (s)",        lambda r: _get(r, 'train_time_seconds')),
        ("Digit Accuracy",        lambda r: _get(r, 'evaluation', 'per_class', 'overall_digit_accuracy')),
        ("Operator Accuracy",     lambda r: _get(r, 'evaluation', 'per_class', 'overall_operator_accuracy')),
        ("Expression Accuracy",   lambda r: _get(r, 'evaluation', 'expression', 'expression_accuracy')),
        ("Result Accuracy",       lambda r: _get(r, 'evaluation', 'result', 'result_accuracy')),
        ("ECE (Calibration)",     lambda r: _get(r, 'evaluation', 'calibration', 'ece')),
        ("Abduction Rate",        lambda r: _get(r, 'evaluation', 'abduction', 'abduction_rate')),
        ("Abd. Effectiveness",    lambda r: _get(r, 'evaluation', 'abduction', 'abduction_effectiveness')),
    ]

    for label, extractor in rows:
        row = f"  {label:<33}"
        for name in configs:
            row += f"  {extractor(results[name]):<{col}}"
        print(row)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=None,
                        help='Path to saved ablations JSON. If omitted, runs experiments first.')
    parser.add_argument('--epochs',  type=int,   default=5)
    parser.add_argument('--samples', type=int,   default=2000)
    parser.add_argument('--batch',   type=int,   default=32)
    parser.add_argument('--lr',      type=float, default=0.01)
    parser.add_argument('--output',  type=str,   default='results/ablations.json')
    args = parser.parse_args()

    if args.results and os.path.exists(args.results):
        print(f"Loading results from: {args.results}")
        with open(args.results) as f:
            results = json.load(f)
    else:
        print("No results file found — running ablations first...")
        from experiments.ablations import run_ablations
        results = run_ablations(
            epochs=args.epochs,
            num_samples=args.samples,
            batch_size=args.batch,
            learning_rate=args.lr,
            output_path=args.output,
        )

    print_table(results)


if __name__ == '__main__':
    main()
