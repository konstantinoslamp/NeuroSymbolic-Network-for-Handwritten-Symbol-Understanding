"""
ablations.py
Run the full ablation study: Pure Neural vs NGA vs WMC.

Usage:
    python experiments/ablations.py
    python experiments/ablations.py --epochs 5 --samples 2000 --output results/ablations.json
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
from src.evaluation.ablation_studies import AblationRunner, ABLATION_CONFIGS, AblationConfig


def run_ablations(epochs=5, num_samples=2000, batch_size=32,
                  learning_rate=0.01, output_path=None):
    print("=" * 60)
    print("  ABLATION STUDY: Pure Neural | NGA | WMC")
    print("=" * 60)

    print("\n[1/3] Building datasets...")
    train_dataset = ExpressionDataset(num_samples=num_samples, split='train', invalid_ratio=0.05)
    test_dataset  = ExpressionDataset(num_samples=500,         split='test',  invalid_ratio=0.05)
    print(f"  Train: {len(train_dataset)} samples  |  Test: {len(test_dataset)} samples")

    # Override epochs / lr / batch_size on the standard configs
    configs = {
        name: AblationConfig(
            name=cfg.name,
            use_symbolic=cfg.use_symbolic,
            abduction_strategy=cfg.abduction_strategy,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        for name, cfg in ABLATION_CONFIGS.items()
    }

    print(f"\n[2/3] Running {len(configs)} ablations × {epochs} epochs each...")
    runner = AblationRunner(
        neural_module_factory=DigitRecognizer,
        symbolic_module_factory=ArithmeticSymbolicModule,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    t0 = time.time()
    results = runner.run_all(configs=configs)
    elapsed = time.time() - t0

    print(f"\n[3/3] All ablations complete in {elapsed:.1f}s")
    AblationRunner.print_comparison(results)

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
    parser.add_argument('--epochs',  type=int,   default=5)
    parser.add_argument('--samples', type=int,   default=2000)
    parser.add_argument('--batch',   type=int,   default=32)
    parser.add_argument('--lr',      type=float, default=0.01)
    parser.add_argument('--output',  type=str,   default='results/ablations.json')
    args = parser.parse_args()

    run_ablations(
        epochs=args.epochs,
        num_samples=args.samples,
        batch_size=args.batch,
        learning_rate=args.lr,
        output_path=args.output,
    )
