"""
Gradient Flow Verification (P1.4)

Monitors and verifies that semantic loss gradients flow correctly
through the CNN during neuro-symbolic training.

Provides:
  - Per-layer gradient norm logging
  - Gradient health checks (zero, exploding, vanishing)
  - Before/after weight histogram snapshots
  - Weight delta analysis (proves training changes the CNN)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class GradientMonitor:
    """
    Monitors gradient flow through the CNN during neuro-symbolic training.

    Attach to the training loop to verify that:
    1. Semantic loss produces non-zero gradients
    2. Gradients are not exploding (norm < threshold)
    3. Gradients are not vanishing (norm > epsilon)
    4. CNN weights actually change during neurosymbolic training
    """

    def __init__(self, exploding_threshold: float = 100.0,
                 vanishing_threshold: float = 1e-7):
        self.exploding_threshold = exploding_threshold
        self.vanishing_threshold = vanishing_threshold

        # History
        self.gradient_norms_history = []    # per-step gradient norms
        self.weight_norms_history = []      # per-step weight norms
        self.gradient_stats_history = []    # per-step gradient statistics

        # Snapshots for before/after comparison
        self.initial_weights = None
        self.weight_snapshots = {}

        # Alerts
        self.alerts = []

    # ------------------------------------------------------------------
    # Weight Snapshots
    # ------------------------------------------------------------------

    def snapshot_weights(self, model, label: str = 'initial'):
        """
        Save a deep copy of all trainable weight matrices.

        Args:
            model: CNN model with trainable_layers attribute
            label: snapshot identifier (e.g., 'initial', 'epoch_1', 'final')
        """
        snapshot = {}
        for i, layer in enumerate(model.trainable_layers):
            layer_name = f'layer_{i}_{type(layer).__name__}'
            snapshot[layer_name] = {
                'W': layer.W.copy(),
                'b': layer.b.copy(),
            }
        self.weight_snapshots[label] = snapshot

        if label == 'initial':
            self.initial_weights = snapshot

    # ------------------------------------------------------------------
    # Per-Step Gradient Monitoring
    # ------------------------------------------------------------------

    def log_gradients(self, model, step: int = None) -> Dict:
        """
        Log gradient norms and statistics for all trainable layers.

        Call AFTER model.backward() and BEFORE model.update_weights().

        Args:
            model: CNN model with trainable_layers
            step: optional step counter

        Returns:
            Dict with gradient statistics per layer
        """
        layer_stats = {}
        all_grad_norms = []

        for i, layer in enumerate(model.trainable_layers):
            layer_name = f'layer_{i}_{type(layer).__name__}'

            if not hasattr(layer, 'grad_W') or layer.grad_W is None:
                layer_stats[layer_name] = {'status': 'no_gradient'}
                self.alerts.append({
                    'step': step,
                    'layer': layer_name,
                    'type': 'missing_gradient',
                    'message': f'No gradient computed for {layer_name}',
                })
                continue

            grad_w = layer.grad_W
            grad_b = layer.grad_b

            # Compute norms
            w_norm = float(np.linalg.norm(grad_w))
            b_norm = float(np.linalg.norm(grad_b))
            total_norm = float(np.sqrt(w_norm**2 + b_norm**2))
            all_grad_norms.append(total_norm)

            stats = {
                'grad_W_norm': w_norm,
                'grad_b_norm': b_norm,
                'total_norm': total_norm,
                'grad_W_mean': float(np.mean(grad_w)),
                'grad_W_std': float(np.std(grad_w)),
                'grad_W_min': float(np.min(grad_w)),
                'grad_W_max': float(np.max(grad_w)),
                'grad_W_nonzero_ratio': float(np.mean(np.abs(grad_w) > 1e-10)),
            }
            layer_stats[layer_name] = stats

            # Check for issues
            if total_norm < self.vanishing_threshold:
                self.alerts.append({
                    'step': step, 'layer': layer_name,
                    'type': 'vanishing_gradient',
                    'message': f'Gradient norm {total_norm:.2e} below threshold',
                })
            elif total_norm > self.exploding_threshold:
                self.alerts.append({
                    'step': step, 'layer': layer_name,
                    'type': 'exploding_gradient',
                    'message': f'Gradient norm {total_norm:.2e} above threshold',
                })

            if np.any(np.isnan(grad_w)) or np.any(np.isinf(grad_w)):
                self.alerts.append({
                    'step': step, 'layer': layer_name,
                    'type': 'nan_inf_gradient',
                    'message': 'NaN or Inf detected in gradients',
                })

        record = {
            'step': step,
            'per_layer': layer_stats,
            'total_grad_norm': float(np.linalg.norm(all_grad_norms)) if all_grad_norms else 0.0,
        }
        self.gradient_norms_history.append(record)
        return record

    def log_weight_norms(self, model, step: int = None) -> Dict:
        """Log weight norms for all trainable layers."""
        layer_norms = {}
        for i, layer in enumerate(model.trainable_layers):
            layer_name = f'layer_{i}_{type(layer).__name__}'
            layer_norms[layer_name] = {
                'W_norm': float(np.linalg.norm(layer.W)),
                'b_norm': float(np.linalg.norm(layer.b)),
            }

        record = {'step': step, 'per_layer': layer_norms}
        self.weight_norms_history.append(record)
        return record

    # ------------------------------------------------------------------
    # Before/After Weight Analysis
    # ------------------------------------------------------------------

    def compute_weight_deltas(self, label_before: str = 'initial',
                               label_after: str = 'final') -> Dict:
        """
        Compare weight snapshots to verify that training changed the CNN.

        Returns per-layer weight deltas (L2 norm of change),
        relative change, and histogram bin counts.
        """
        if label_before not in self.weight_snapshots:
            return {'error': f'No snapshot with label "{label_before}"'}
        if label_after not in self.weight_snapshots:
            return {'error': f'No snapshot with label "{label_after}"'}

        before = self.weight_snapshots[label_before]
        after = self.weight_snapshots[label_after]

        deltas = {}
        for layer_name in before:
            w_before = before[layer_name]['W']
            w_after = after[layer_name]['W']
            b_before = before[layer_name]['b']
            b_after = after[layer_name]['b']

            w_delta = w_after - w_before
            b_delta = b_after - b_before

            w_norm_before = np.linalg.norm(w_before)
            w_delta_norm = np.linalg.norm(w_delta)

            # Weight histograms (before and after)
            w_hist_before, w_bins = np.histogram(w_before.flatten(), bins=50)
            w_hist_after, _ = np.histogram(w_after.flatten(), bins=w_bins)

            deltas[layer_name] = {
                'W_delta_norm': float(w_delta_norm),
                'W_relative_change': float(w_delta_norm / (w_norm_before + 1e-10)),
                'b_delta_norm': float(np.linalg.norm(b_delta)),
                'W_delta_mean': float(np.mean(np.abs(w_delta))),
                'W_delta_max': float(np.max(np.abs(w_delta))),
                'weights_changed': bool(w_delta_norm > 1e-10),
                'histogram_before': w_hist_before.tolist(),
                'histogram_after': w_hist_after.tolist(),
                'histogram_bins': w_bins.tolist(),
            }

        return deltas

    # ------------------------------------------------------------------
    # Sanity Checks
    # ------------------------------------------------------------------

    def run_sanity_checks(self, model) -> Dict:
        """
        Run a battery of sanity checks on the current model state.

        Returns a dict of check_name -> (passed: bool, message: str).
        """
        checks = {}

        # Check 1: Weights are finite
        all_finite = True
        for i, layer in enumerate(model.trainable_layers):
            if np.any(np.isnan(layer.W)) or np.any(np.isinf(layer.W)):
                all_finite = False
                break
            if np.any(np.isnan(layer.b)) or np.any(np.isinf(layer.b)):
                all_finite = False
                break
        checks['weights_finite'] = {
            'passed': all_finite,
            'message': 'All weights are finite' if all_finite else 'NaN/Inf in weights!',
        }

        # Check 2: Gradients were computed (non-zero)
        has_gradients = False
        for layer in model.trainable_layers:
            if hasattr(layer, 'grad_W') and layer.grad_W is not None:
                if np.linalg.norm(layer.grad_W) > 1e-10:
                    has_gradients = True
                    break
        checks['nonzero_gradients'] = {
            'passed': has_gradients,
            'message': 'Non-zero gradients present' if has_gradients
                       else 'WARNING: All gradients are zero!',
        }

        # Check 3: Weights changed from initial (if snapshot exists)
        if self.initial_weights:
            weights_changed = False
            for i, layer in enumerate(model.trainable_layers):
                layer_name = f'layer_{i}_{type(layer).__name__}'
                if layer_name in self.initial_weights:
                    delta = np.linalg.norm(layer.W - self.initial_weights[layer_name]['W'])
                    if delta > 1e-10:
                        weights_changed = True
                        break
            checks['weights_changed'] = {
                'passed': weights_changed,
                'message': 'Weights changed from initial values' if weights_changed
                           else 'WARNING: Weights unchanged from initialization!',
            }

        # Check 4: No gradient explosion in recent history
        if self.gradient_norms_history:
            recent_norms = [r['total_grad_norm'] for r in self.gradient_norms_history[-10:]]
            no_explosion = all(n < self.exploding_threshold for n in recent_norms)
            checks['no_explosion'] = {
                'passed': no_explosion,
                'message': 'No gradient explosion detected' if no_explosion
                           else f'WARNING: Gradient norms exceeded {self.exploding_threshold}!',
            }

        return checks

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_gradient_report(self):
        """Print a summary of gradient flow diagnostics."""
        print("\n" + "=" * 60)
        print("  GRADIENT FLOW REPORT")
        print("=" * 60)

        # Recent gradient norms
        if self.gradient_norms_history:
            recent = self.gradient_norms_history[-5:]
            print("\n--- Recent Gradient Norms ---")
            for record in recent:
                step = record.get('step', '?')
                print(f"  Step {step}: total_norm = {record['total_grad_norm']:.6f}")
                for layer, stats in record['per_layer'].items():
                    if isinstance(stats, dict) and 'total_norm' in stats:
                        print(f"    {layer}: norm={stats['total_norm']:.6f}, "
                              f"nonzero={stats['grad_W_nonzero_ratio']:.2%}")

        # Weight deltas
        if 'initial' in self.weight_snapshots and 'final' in self.weight_snapshots:
            deltas = self.compute_weight_deltas('initial', 'final')
            print("\n--- Weight Changes (initial -> final) ---")
            for layer, d in deltas.items():
                if isinstance(d, dict) and 'W_delta_norm' in d:
                    print(f"  {layer}:")
                    print(f"    Delta norm:     {d['W_delta_norm']:.6f}")
                    print(f"    Relative change: {d['W_relative_change']:.4%}")
                    print(f"    Changed: {d['weights_changed']}")

        # Alerts
        if self.alerts:
            print(f"\n--- Alerts ({len(self.alerts)} total) ---")
            for alert in self.alerts[-10:]:
                print(f"  [{alert['type']}] Step {alert.get('step', '?')}: "
                      f"{alert['message']}")

        print("=" * 60)

    def summary(self) -> Dict:
        """Return structured summary for programmatic access."""
        grad_norms = [r['total_grad_norm'] for r in self.gradient_norms_history]

        return {
            'num_steps_logged': len(self.gradient_norms_history),
            'avg_grad_norm': float(np.mean(grad_norms)) if grad_norms else 0.0,
            'max_grad_norm': float(np.max(grad_norms)) if grad_norms else 0.0,
            'min_grad_norm': float(np.min(grad_norms)) if grad_norms else 0.0,
            'num_alerts': len(self.alerts),
            'alert_types': list(set(a['type'] for a in self.alerts)),
            'weight_snapshots': list(self.weight_snapshots.keys()),
        }
