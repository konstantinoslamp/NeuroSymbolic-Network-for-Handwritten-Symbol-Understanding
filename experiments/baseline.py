"""
baseline.py
Run the pure neural baseline (no symbolic feedback) and save results.

Usage:
    python experiments/baseline.py
    python experiments/baseline.py --epochs 5 --samples 2000 --output results/baseline.json
"""

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.expression_dataset import ExpressionDataset
from src.neural.digit_recognizer import DigitRecognizer
from src.symbolic.symbolic_interface import ArithmeticSymbolicModule
from src.evaluation.ablation_studies import AblationRunner, AblationConfig


def run_baseline(epochs=5, num_samples=2000, batch_size=32,
                 learning_rate=0.01, output_path=None):
    print("=" * 60)
    print("  PURE NEURAL BASELINE")
    print("=" * 60)

    print("\n[1/3] Building datasets...")
    train_dataset = ExpressionDataset(num_samples=num_samples, split='train', invalid_ratio=0.05)
    test_dataset  = ExpressionDataset(num_samples=500,         split='test',  invalid_ratio=0.05)
    print(f"  Train: {len(train_dataset)} samples  |  Test: {len(test_dataset)} samples")

    config = AblationConfig(
        name='Pure Neural (No Symbolic)',
        use_symbolic=False,
        abduction_strategy='none',
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )

    print("\n[2/3] Training...")
    runner = AblationRunner(
        neural_module_factory=DigitRecognizer,
        symbolic_module_factory=ArithmeticSymbolicModule,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    t0 = time.time()
    results = runner.run_all(configs={'pure_neural': config})
    elapsed = time.time() - t0

    print(f"\n[3/3] Done in {elapsed:.1f}s")

    res = results['pure_neural']
    print("\n--- Results ---")
    for ep in res['training_history']:
        print(f"  Epoch {ep['epoch']}: loss={ep['avg_loss']:.4f}  acc={ep['accuracy']:.4f}")

    eval_r = res.get('evaluation', {})
    expr   = eval_r.get('expression', {})
    result = eval_r.get('result', {})
    print(f"\n  Expression accuracy : {expr.get('expression_accuracy', 'N/A')}")
    print(f"  Result accuracy     : {result.get('result_accuracy', 'N/A')}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        def _convert(obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_convert(v) for v in obj]
            return obj
        with open(output_path, 'w') as f:
            json.dump(_convert(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',   type=int,   default=5)
    parser.add_argument('--samples',  type=int,   default=2000)
    parser.add_argument('--batch',    type=int,   default=32)
    parser.add_argument('--lr',       type=float, default=0.01)
    parser.add_argument('--output',   type=str,   default='results/baseline.json')
    args = parser.parse_args()

    run_baseline(
        epochs=args.epochs,
        num_samples=args.samples,
        batch_size=args.batch,
        learning_rate=args.lr,
        output_path=args.output,
    )
