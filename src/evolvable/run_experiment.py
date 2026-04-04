"""
Main Experiment Runner for Evolvable Policies

End-to-end pipeline:
  1. Generate a random hidden target policy
  2. Create MNIST-based dataset from the target policy
  3. Run the evolutionary process
  4. Report results and save logs

Usage:
    python -m src.evolvable.run_experiment [--num_atoms 8] [--num_rules 5]
        [--generations 50] [--train_epochs 3] [--seed 42]
"""

import argparse
import json
import os
import time
import numpy as np
from typing import Any, Dict, List

from src.evolvable.machine_coaching import PolicyGenerator
from src.evolvable.dataset import create_experiment_data
from src.evolvable.evolution import EvolutionaryEngine, summarize_evolution
from src.evolvable.organism import NeSyOrganism


def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single evolvable policies experiment.

    Args:
        config: {
            'num_atoms': int (default 8),
            'num_rules': int (default 5),
            'policy_seed': int (default 42),
            'data_seed': int (default 123),
            'train_size': int (default 300),
            'val_size': int (default 100),
            'test_size': int (default 100),
            'train_epochs': int (default 3),
            'learning_rate': float (default 0.001),
            'max_generations': int (default 50),
            'early_stop_accuracy': float (default 0.99),
            'mnist_path': str or None,
            'verbose': bool (default True),
        }

    Returns:
        Full results dict with evolution history, final metrics, etc.
    """
    num_atoms = config.get('num_atoms', 8)
    num_rules = config.get('num_rules', 5)
    policy_seed = config.get('policy_seed', 42)
    data_seed = config.get('data_seed', 123)
    train_size = config.get('train_size', 300)
    val_size = config.get('val_size', 100)
    test_size = config.get('test_size', 100)
    train_epochs = config.get('train_epochs', 3)
    learning_rate = config.get('learning_rate', 0.001)
    max_generations = config.get('max_generations', 50)
    early_stop_accuracy = config.get('early_stop_accuracy', 0.99)
    mnist_path = config.get('mnist_path', None)
    verbose = config.get('verbose', True)

    atoms = [f"a{i+1}" for i in range(num_atoms)]

    # Step 1: Generate target policy and dataset
    if verbose:
        print("=" * 60)
        print("EVOLVABLE POLICIES EXPERIMENT")
        print("=" * 60)
        print(f"Atoms: {num_atoms}, Rules: {num_rules}, Seed: {policy_seed}")
        print(f"Dataset: train={train_size}, val={val_size}, test={test_size}")
        print(f"Evolution: generations={max_generations}, "
              f"train_epochs={train_epochs}, lr={learning_rate}")
        print()

    if verbose:
        print("Generating target policy and dataset...")

    data = create_experiment_data(
        num_atoms=num_atoms,
        num_rules=num_rules,
        policy_seed=policy_seed,
        data_seed=data_seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        mnist_path=mnist_path,
    )

    target_policy = data['target_policy']
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']

    if verbose:
        print(f"Target policy:\n{target_policy}")
        print(f"\nDataset sizes: train={len(train_data)}, "
              f"val={len(val_data)}, test={len(test_data)}")

        # Label distribution
        train_pos = sum(1 for s in train_data if s['label'])
        val_pos = sum(1 for s in val_data if s['label'])
        print(f"Label balance: train={train_pos}/{len(train_data)} positive, "
              f"val={val_pos}/{len(val_data)} positive")
        print()

    # Step 2: Run evolution
    if verbose:
        print("Starting evolutionary process...")
        print("-" * 60)

    start_time = time.time()

    engine = EvolutionaryEngine(
        atoms=atoms,
        train_epochs=train_epochs,
        learning_rate=learning_rate,
        max_generations=max_generations,
        early_stop_accuracy=early_stop_accuracy,
        verbose=verbose,
    )

    evolution_result = engine.run(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )

    total_time = time.time() - start_time

    # Step 3: Summarize
    summary = summarize_evolution(evolution_result['history'])

    if verbose:
        print()
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Generations: {summary['num_generations']}")
        print(f"Initial accuracy: {summary['initial_accuracy']:.3f}")
        print(f"Final accuracy:   {summary['final_accuracy']:.3f}")
        print(f"Best accuracy:    {summary['best_accuracy']:.3f}")
        print(f"Final wrong rate: {summary['final_wrong_rate']:.3f}")
        print(f"Final abstain:    {summary['final_abstain_rate']:.3f}")
        print(f"Selection counts: {summary['selection_counts']}")
        print(f"Total time: {total_time:.1f}s")

        if 'final_test' in evolution_result:
            test = evolution_result['final_test']
            print(f"\nTest set: correct={test['correct']:.3f} "
                  f"wrong={test['wrong']:.3f} "
                  f"abstain={test['abstain']:.3f}")

        # Lineage
        if evolution_result['lineage']:
            print(f"\nLineage (last 10 generations):")
            for gen, label, fitness in evolution_result['lineage'][-10:]:
                print(f"  Gen {gen}: {label} (fitness={fitness:+.3f})")

    return {
        'config': config,
        'target_policy': str(target_policy),
        'summary': summary,
        'evolution_result': {
            'best_val_accuracy': evolution_result['best_val_accuracy'],
            'final_test': evolution_result.get('final_test'),
            'lineage': evolution_result['lineage'],
        },
        'total_time': total_time,
    }


def run_batch_experiments(num_policies: int = 5,
                          base_seed: int = 42,
                          **kwargs) -> List:
    """
    Run experiments over multiple random target policies.

    Args:
        num_policies: number of different target policies to test
        base_seed: starting seed
        **kwargs: passed to run_single_experiment config

    Returns:
        List of result dicts
    """
    results = []

    for i in range(num_policies):
        print(f"\n{'#' * 60}")
        print(f"# EXPERIMENT {i+1}/{num_policies} (seed={base_seed + i})")
        print(f"{'#' * 60}\n")

        config = {
            'policy_seed': base_seed + i,
            'data_seed': base_seed + i + 1000,
            **kwargs,
        }

        result = run_single_experiment(config)
        results.append(result)

    # Aggregate
    accuracies = [r['summary']['final_accuracy'] for r in results]
    best_accs = [r['summary']['best_accuracy'] for r in results]

    print(f"\n{'=' * 60}")
    print(f"BATCH SUMMARY ({num_policies} policies)")
    print(f"{'=' * 60}")
    print(f"Mean final accuracy: {np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}")
    print(f"Mean best accuracy:  {np.mean(best_accs):.3f} +/- {np.std(best_accs):.3f}")
    print(f"Min/Max final: {np.min(accuracies):.3f} / {np.max(accuracies):.3f}")

    return results


def save_results(results: Any, path: str):
    """Save results to JSON file."""
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run Evolvable Policies experiment'
    )
    parser.add_argument('--num_atoms', type=int, default=8)
    parser.add_argument('--num_rules', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generations', type=int, default=50)
    parser.add_argument('--train_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_size', type=int, default=300)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--batch', type=int, default=0,
                        help='Run batch experiments over N policies (0=single)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    config = {
        'num_atoms': args.num_atoms,
        'num_rules': args.num_rules,
        'policy_seed': args.seed,
        'data_seed': args.seed + 1000,
        'train_size': args.train_size,
        'val_size': args.val_size,
        'test_size': args.test_size,
        'train_epochs': args.train_epochs,
        'learning_rate': args.lr,
        'max_generations': args.generations,
    }

    if args.batch > 0:
        results = run_batch_experiments(
            num_policies=args.batch,
            base_seed=args.seed,
            **{k: v for k, v in config.items()
               if k not in ('policy_seed', 'data_seed')},
        )
    else:
        results = run_single_experiment(config)

    if args.output:
        save_results(results, args.output)


if __name__ == '__main__':
    main()
