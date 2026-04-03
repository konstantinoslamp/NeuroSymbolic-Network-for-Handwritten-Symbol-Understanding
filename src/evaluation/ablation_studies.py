"""
Ablation Studies Framework

Compares:
  1. Pure Neural Baseline (no symbolic feedback)
  2. NGA (Neural-Guided Abduction) - single best path
  3. WMC (Weighted Model Counting) - full probabilistic semantic loss

Each ablation trains from the same initial weights and evaluates
on the same held-out test set for fair comparison.
"""

import numpy as np
import copy
import time
from typing import Dict, List, Optional
from src.evaluation.metrics import EvaluationSuite


class AblationConfig:
    """Configuration for a single ablation run."""

    def __init__(self, name: str, use_symbolic: bool = True,
                 abduction_strategy: str = 'wmc', epochs: int = 5,
                 learning_rate: float = 0.001, batch_size: int = 32):
        self.name = name
        self.use_symbolic = use_symbolic
        self.abduction_strategy = abduction_strategy
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size


# Standard ablation configurations
ABLATION_CONFIGS = {
    'pure_neural': AblationConfig(
        name='Pure Neural (No Symbolic)',
        use_symbolic=False,
        abduction_strategy='none',
    ),
    'nga': AblationConfig(
        name='NGA (Neural-Guided Abduction)',
        use_symbolic=True,
        abduction_strategy='nga',
    ),
    'wmc': AblationConfig(
        name='WMC (Weighted Model Counting)',
        use_symbolic=True,
        abduction_strategy='wmc',
    ),
}


class AblationRunner:
    """
    Runs ablation studies comparing different training strategies.

    Each configuration is trained from the same initial weights
    and evaluated on the same test set.
    """

    def __init__(self, neural_module_factory, symbolic_module_factory,
                 train_dataset, test_dataset):
        """
        Args:
            neural_module_factory: callable that returns a fresh NeuralModule
            symbolic_module_factory: callable that returns a fresh SymbolicModule
            train_dataset: training dataset
            test_dataset: test/evaluation dataset
        """
        self.neural_factory = neural_module_factory
        self.symbolic_factory = symbolic_module_factory
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.results = {}

    def run_all(self, configs: Dict[str, AblationConfig] = None) -> Dict:
        """
        Run all ablation configurations.

        Args:
            configs: dict of name -> AblationConfig. Defaults to standard set.

        Returns:
            Dict mapping config name to evaluation results + training history.
        """
        if configs is None:
            configs = ABLATION_CONFIGS

        # Save initial weights for reproducibility
        reference_neural = self.neural_factory()
        initial_weights = reference_neural.get_parameters()

        for config_name, config in configs.items():
            print(f"\n{'='*60}")
            print(f"  ABLATION: {config.name}")
            print(f"{'='*60}")

            result = self._run_single(config, initial_weights)
            self.results[config_name] = result

        return self.results

    def _run_single(self, config: AblationConfig,
                    initial_weights: Dict) -> Dict:
        """Run a single ablation configuration."""
        neural = self.neural_factory()
        neural.set_parameters(copy.deepcopy(initial_weights))
        symbolic = self.symbolic_factory()

        training_history = []
        start_time = time.time()

        if config.use_symbolic:
            from src.integration.training_loop import NeuroSymbolicTrainer
            trainer = NeuroSymbolicTrainer(neural, symbolic, config)

            for epoch in range(config.epochs):
                epoch_metrics = self._train_epoch_symbolic(
                    trainer, config.batch_size
                )
                epoch_metrics['epoch'] = epoch + 1
                training_history.append(epoch_metrics)
                print(f"  Epoch {epoch+1}/{config.epochs} - "
                      f"Loss: {epoch_metrics['avg_loss']:.4f}, "
                      f"Acc: {epoch_metrics['accuracy']:.4f}")
        else:
            # Pure neural baseline: standard cross-entropy training
            for epoch in range(config.epochs):
                epoch_metrics = self._train_epoch_pure_neural(
                    neural, config.batch_size, config.learning_rate
                )
                epoch_metrics['epoch'] = epoch + 1
                training_history.append(epoch_metrics)
                print(f"  Epoch {epoch+1}/{config.epochs} - "
                      f"Loss: {epoch_metrics['avg_loss']:.4f}, "
                      f"Acc: {epoch_metrics['accuracy']:.4f}")

        train_time = time.time() - start_time

        # Evaluate on test set
        print(f"  Evaluating on test set...")
        eval_suite = EvaluationSuite()
        eval_results = eval_suite.evaluate(neural, symbolic, self.test_dataset)

        return {
            'config': config.name,
            'strategy': config.abduction_strategy,
            'training_history': training_history,
            'evaluation': eval_results,
            'train_time_seconds': train_time,
        }

    def _train_epoch_symbolic(self, trainer, batch_size: int) -> Dict:
        """Train one epoch using neuro-symbolic loop."""
        indices = np.arange(len(self.train_dataset))
        np.random.shuffle(indices)

        total_loss = 0.0
        total_correct = 0
        total_abductions = 0
        num_batches = 0

        for start in range(0, len(self.train_dataset), batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) == 0:
                break

            images = []
            results = []
            for idx in batch_idx:
                item = self.train_dataset[idx]
                images.append(item['images'])
                results.append(item['result'])

            images = np.stack(images)
            metrics = trainer.train_step(images, results)

            total_loss += metrics['loss']
            total_correct += metrics['correct']
            total_abductions += metrics['abductions']
            num_batches += 1

        return {
            'avg_loss': total_loss / max(num_batches, 1),
            'accuracy': total_correct / len(self.train_dataset),
            'abductions': total_abductions,
        }

    def _train_epoch_pure_neural(self, neural, batch_size: int,
                                  learning_rate: float) -> Dict:
        """Train one epoch using pure neural cross-entropy (no symbolic feedback)."""
        symbol_to_idx = {str(i): i for i in range(10)}
        symbol_to_idx.update({'+': 10, '-': 11, '×': 12, '÷': 13})

        indices = np.arange(len(self.train_dataset))
        np.random.shuffle(indices)

        total_loss = 0.0
        total_correct = 0
        num_batches = 0

        for start in range(0, len(self.train_dataset), batch_size):
            batch_idx = indices[start:start + batch_size]
            if len(batch_idx) == 0:
                break

            all_images = []
            all_labels = []

            for idx in batch_idx:
                item = self.train_dataset[idx]
                text = item.get('text', '')
                if isinstance(text, str) and len(text) >= 3:
                    syms = [text[0], text[1], text[2]]
                elif isinstance(text, list):
                    syms = text
                else:
                    continue

                for i, sym in enumerate(syms):
                    if sym in symbol_to_idx:
                        img = item['images'][i]
                        all_images.append(img[np.newaxis, :, :])
                        all_labels.append(symbol_to_idx[sym])

            if not all_images:
                continue

            x = np.stack(all_images)
            y = np.array(all_labels)

            logits = neural.model.forward(x)
            loss = neural.model.compute_loss(logits, y)
            neural.model.backward()
            neural.model.update_weights(learning_rate)

            preds = np.argmax(logits, axis=1)
            total_correct += np.sum(preds == y)
            total_loss += float(loss)
            num_batches += 1

        total_samples = len(self.train_dataset) * 3  # 3 symbols per expression
        return {
            'avg_loss': total_loss / max(num_batches, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'abductions': 0,
        }

    @staticmethod
    def print_comparison(results: Dict):
        """Print a side-by-side comparison of ablation results."""
        print("\n" + "=" * 80)
        print("  ABLATION STUDY COMPARISON")
        print("=" * 80)

        header = f"{'Metric':<35}"
        for name in results:
            header += f" {results[name]['config']:<20}"
        print(header)
        print("-" * 80)

        # Training metrics (final epoch)
        metrics_to_compare = [
            ('Final Train Loss', lambda r: r['training_history'][-1]['avg_loss']),
            ('Final Train Accuracy', lambda r: r['training_history'][-1]['accuracy']),
            ('Training Time (s)', lambda r: r['train_time_seconds']),
        ]

        # Evaluation metrics
        eval_metrics = [
            ('Digit Accuracy', lambda r: r['evaluation']['per_class']['overall_digit_accuracy']),
            ('Operator Accuracy', lambda r: r['evaluation']['per_class']['overall_operator_accuracy']),
            ('Expression Accuracy', lambda r: r['evaluation']['expression']['expression_accuracy']),
            ('Result Accuracy', lambda r: r['evaluation']['result']['result_accuracy']),
            ('ECE (Calibration)', lambda r: r['evaluation']['calibration']['ece']),
            ('Abduction Rate', lambda r: r['evaluation']['abduction']['abduction_rate']),
            ('Abd. Effectiveness', lambda r: r['evaluation']['abduction']['abduction_effectiveness']),
        ]

        for label, extractor in metrics_to_compare + eval_metrics:
            row = f"  {label:<33}"
            for name in results:
                try:
                    val = extractor(results[name])
                    row += f" {val:<20.4f}"
                except (KeyError, IndexError, TypeError):
                    row += f" {'N/A':<20}"
            print(row)

        print("=" * 80)
