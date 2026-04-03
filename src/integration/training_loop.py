"""
Neuro-symbolic Training Loop (The 5-Step Cycle)

Uses WMC-based semantic loss computed via d-DNNF arithmetic circuits
(Xu et al. 2018, Tsamoura et al. AAAI 2021).
"""

import numpy as np
from typing import Dict, List, Any
import time

from src.integration.semantic_loss import SemanticLossWMC
from src.utils.gradient_monitor import GradientMonitor


class NeuroSymbolicTrainer:
    def __init__(self, neural_module, symbolic_module, task_config):
        self.neural = neural_module
        self.symbolic = symbolic_module
        self.config = task_config
        self.history = []

        # WMC-based semantic loss engine
        self.semantic_loss = SemanticLossWMC(num_classes=14)

        # Gradient flow monitor (P1.4)
        self.gradient_monitor = GradientMonitor()
        self.gradient_monitor.snapshot_weights(neural_module.model, 'initial')
        self._step_counter = 0

        # Mapping symbols to indices (0-9 digits, 10-13 operators)
        self.symbol_to_idx = {str(i): i for i in range(10)}
        self.symbol_to_idx.update({'+': 10, '-': 11, '×': 12, '÷': 13})
        self.idx_to_symbol = {v: k for k, v in self.symbol_to_idx.items()}

    def train_step(self, images: np.ndarray, ground_truth_results: List[float]) -> Dict:
        """
        Executes one iteration of the neuro-symbolic loop.
        Supports two strategies via self.config.abduction_strategy:
        1. 'nga': Neural-Guided Abduction (Maximum Likelihood) - Picks the single best path.
        2. 'wmc': Weighted Model Counting (Semantic Loss) - Exact d-DNNF circuit WMC.
        """
        batch_size = len(images)
        metrics = {'correct': 0, 'abductions': 0, 'loss': 0.0, 'wmc_values': []}
        strategy = getattr(self.config, 'abduction_strategy', 'wmc')

        # --- STEP 1: Neural Deduction (Forward Pass) ---
        neural_out = self.neural.neural_deduction(images)
        probs = neural_out['probabilities']  # (Batch, Seq, Classes)

        # Initialize gradients for the whole batch
        gradients = np.zeros_like(probs)  # (Batch, Seq, Classes)

        predicted_symbols_batch = self._decode_neural_output(neural_out)
        total_loss = 0.0

        for i in range(batch_size):
            symbols = predicted_symbols_batch[i]
            target_result = ground_truth_results[i]
            sample_probs = probs[i]  # (Seq, Classes)

            # --- STEP 2: Symbolic Deduction ---
            deduction = self.symbolic.symbolic_deduction({'symbols': symbols})

            is_correct = False
            if target_result is None:
                if not deduction['valid'] or deduction['result'] is None:
                    is_correct = True
                    metrics['correct'] += 1
            elif deduction['valid'] and deduction['result'] is not None:
                if abs(deduction['result'] - target_result) < 0.01:
                    is_correct = True
                    metrics['correct'] += 1

            # --- STEP 3: Abduction & WMC-based Semantic Loss ---
            if target_result is not None:
                valid_paths = self.symbolic.symbolic_abduction(
                    desired_output=target_result,
                    current_state={'symbols': symbols},
                    neural_probs={}
                )

                if valid_paths:
                    metrics['abductions'] += 1

                    # Compute semantic loss via d-DNNF circuit WMC
                    loss, grad = self.semantic_loss.compute_loss_and_gradient(
                        sample_probs, valid_paths, strategy
                    )

                    total_loss += loss
                    gradients[i] = grad

            else:
                # For invalid targets, skip semantic loss
                pass

        metrics['loss'] = total_loss / batch_size if batch_size > 0 else 0

        # --- STEP 4: Neural Induction (Backprop) ---
        self.neural.train_with_gradient(images, gradients)

        # --- Gradient Flow Monitoring (P1.4) ---
        self._step_counter += 1
        self.gradient_monitor.log_gradients(self.neural.model, self._step_counter)
        self.gradient_monitor.log_weight_norms(self.neural.model, self._step_counter)

        return metrics

    def _decode_neural_output(self, neural_out):
        """Helper to convert neural logits/probs to symbol lists"""
        if 'class_ids' in neural_out:
            ids = neural_out['class_ids']
            return [[self.idx_to_symbol.get(idx, '?') for idx in row] for row in ids]
        return []

    def _extract_sample_probs(self, neural_out, index):
        """Helper to get probabilities for a single sample from batch"""
        # neural_out['probabilities'] is expected to be (batch, seq_len, num_classes)
        if 'probabilities' not in neural_out:
            return {}
        
        probs = neural_out['probabilities'][index] # Shape (seq_len, num_classes)
        
        # Convert to dictionary format expected by symbolic_abduction
        # keys: 'position_0', 'position_1', ...
        return {f'position_{i}': row for i, row in enumerate(probs)}