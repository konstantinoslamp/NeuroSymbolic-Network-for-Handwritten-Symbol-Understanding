"""
Uncertainty Quantification for Neural Module — P3.2

Calibrates neural confidence estimates so that P(correct | confidence = p) ≈ p.
This is critical for meaningful abduction: uncalibrated softmax scores are
typically overconfident, distorting P(explanation) scores.

Implements:
  1. Temperature Scaling (Guo et al. 2017) — post-hoc single-parameter calibration
  2. Monte Carlo Dropout — epistemic uncertainty via stochastic forward passes
  3. Calibration diagnostics (ECE, reliability diagrams)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from src.neural.model import CNN


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaling:
    """
    Post-hoc calibration via a single learned temperature parameter T.

    Calibrated probabilities: p_cal = softmax(logits / T)

    T > 1 softens the distribution (reduces overconfidence)
    T < 1 sharpens the distribution (increases confidence)
    T = 1 is the uncalibrated baseline

    Optimization: minimize NLL on a held-out validation set using grid search
    (exact for single parameter; no need for gradient-based optimization).
    """

    def __init__(self):
        self.temperature = 1.0  # Uncalibrated default
        self._fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            t_range: Tuple[float, float] = (0.1, 10.0),
            num_steps: int = 100) -> float:
        """
        Find optimal temperature on validation data.

        Args:
            logits: (N, num_classes) raw logits from CNN
            labels: (N,) ground truth class indices
            t_range: search range for temperature
            num_steps: grid search resolution

        Returns:
            Optimal temperature value
        """
        best_t = 1.0
        best_nll = float('inf')

        temperatures = np.linspace(t_range[0], t_range[1], num_steps)

        for t in temperatures:
            nll = self._compute_nll(logits, labels, t)
            if nll < best_nll:
                best_nll = nll
                best_t = t

        # Fine-grained search around best
        fine_range = (max(t_range[0], best_t - 0.5), min(t_range[1], best_t + 0.5))
        fine_temps = np.linspace(fine_range[0], fine_range[1], num_steps)
        for t in fine_temps:
            nll = self._compute_nll(logits, labels, t)
            if nll < best_nll:
                best_nll = nll
                best_t = t

        self.temperature = best_t
        self._fitted = True
        return best_t

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.

        Args:
            logits: (N, num_classes) or (N, seq_len, num_classes)

        Returns:
            Calibrated probabilities (same shape as input but with softmax applied)
        """
        scaled = logits / self.temperature
        # Stable softmax
        if scaled.ndim == 2:
            shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
            exp_scores = np.exp(shifted)
            return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        elif scaled.ndim == 3:
            shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
            exp_scores = np.exp(shifted)
            return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return scaled

    def _compute_nll(self, logits: np.ndarray, labels: np.ndarray,
                     temperature: float) -> float:
        """Compute negative log-likelihood at given temperature."""
        probs = self.calibrate_at_temp(logits, temperature)
        n = len(labels)
        log_probs = np.log(probs[np.arange(n), labels] + 1e-10)
        return -np.mean(log_probs)

    @staticmethod
    def calibrate_at_temp(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Softmax at specific temperature."""
        scaled = logits / temperature
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    def summary(self) -> Dict:
        return {
            'temperature': self.temperature,
            'fitted': self._fitted,
        }


# ---------------------------------------------------------------------------
# Monte Carlo Dropout
# ---------------------------------------------------------------------------

class MCDropout:
    """
    Monte Carlo Dropout for epistemic uncertainty estimation.

    Instead of using dropout only during training, we keep it active
    during inference and perform T stochastic forward passes.

    The variance across passes captures model uncertainty (epistemic),
    while the mean provides a better-calibrated prediction.

    Predictive distribution:
      p(y|x) ≈ (1/T) Σ_t softmax(f_θ_t(x))

    Uncertainty metrics:
      - Predictive entropy: H[p(y|x)]
      - Mutual information: H[p(y|x)] - E_t[H[p(y|x,θ_t)]]
        (captures epistemic uncertainty specifically)
    """

    def __init__(self, dropout_rate: float = 0.2, num_passes: int = 20):
        self.dropout_rate = dropout_rate
        self.num_passes = num_passes

    def predict_with_uncertainty(self, model: CNN,
                                 images: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform MC Dropout inference.

        Args:
            model: CNN model
            images: (N, 1, 28, 28) batch of images

        Returns:
            dict with mean_probs, std_probs, predictive_entropy,
            mutual_information, class_ids, confidence
        """
        all_probs = []

        for _ in range(self.num_passes):
            # Forward pass with dropout
            logits = self._forward_with_dropout(model, images)

            # Softmax
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            all_probs.append(probs)

        # Stack: (T, N, C)
        all_probs = np.stack(all_probs, axis=0)

        # Mean prediction (better calibrated)
        mean_probs = np.mean(all_probs, axis=0)  # (N, C)

        # Standard deviation (uncertainty per class)
        std_probs = np.std(all_probs, axis=0)  # (N, C)

        # Predictive entropy: H[p(y|x)] = -Σ_c p_c log p_c
        pred_entropy = -np.sum(
            mean_probs * np.log(mean_probs + 1e-10), axis=1
        )  # (N,)

        # Per-pass entropies
        per_pass_entropy = -np.sum(
            all_probs * np.log(all_probs + 1e-10), axis=2
        )  # (T, N)

        # Expected entropy: E_t[H[p(y|x,θ_t)]]
        expected_entropy = np.mean(per_pass_entropy, axis=0)  # (N,)

        # Mutual information (epistemic uncertainty)
        mutual_info = pred_entropy - expected_entropy  # (N,)

        # Best predictions
        class_ids = np.argmax(mean_probs, axis=1)
        confidence = np.max(mean_probs, axis=1)

        return {
            'mean_probs': mean_probs,
            'std_probs': std_probs,
            'predictive_entropy': pred_entropy,
            'mutual_information': mutual_info,
            'expected_entropy': expected_entropy,
            'class_ids': class_ids,
            'confidence': confidence,
            'all_probs': all_probs,
        }

    def _forward_with_dropout(self, model: CNN,
                               images: np.ndarray) -> np.ndarray:
        """
        Forward pass with dropout applied to hidden layers.

        Applies Bernoulli dropout masks to:
          - After conv layer output
          - After first dense layer (fc1)
        """
        x = model.conv1.forward(images)
        x = model.relu1.forward(x)
        x = model.pool1.forward(x)

        # Dropout after conv
        mask_conv = (np.random.random(x.shape) > self.dropout_rate).astype(np.float32)
        x = x * mask_conv / (1.0 - self.dropout_rate)  # Inverted dropout

        x = model.flatten.forward(x)
        x = model.fc1.forward(x)
        x = model.relu2.forward(x)

        # Dropout after fc1
        mask_fc = (np.random.random(x.shape) > self.dropout_rate).astype(np.float32)
        x = x * mask_fc / (1.0 - self.dropout_rate)

        logits = model.fc2.forward(x)
        return logits


# ---------------------------------------------------------------------------
# Calibrated Neural Module Wrapper
# ---------------------------------------------------------------------------

class CalibratedNeuralModule:
    """
    Wraps a neural module with calibration and uncertainty quantification.

    Combines:
      - Temperature scaling for calibrated softmax
      - MC Dropout for uncertainty estimation
      - Calibration diagnostics

    The calibrated probabilities are fed to symbolic abduction,
    yielding meaningful P(explanation) scores.
    """

    def __init__(self, neural_module, dropout_rate: float = 0.2,
                 mc_passes: int = 20):
        """
        Args:
            neural_module: a NeuralModule (e.g., DigitRecognizer)
            dropout_rate: MC Dropout rate
            mc_passes: number of stochastic forward passes
        """
        self.neural = neural_module
        self.temp_scaling = TemperatureScaling()
        self.mc_dropout = MCDropout(dropout_rate, mc_passes)
        self._calibrated = False

    def calibrate(self, val_images: np.ndarray, val_labels: np.ndarray):
        """
        Calibrate using a validation set.

        Args:
            val_images: (N, 1, 28, 28) validation images
            val_labels: (N,) ground truth labels
        """
        # Get raw logits
        logits = self.neural.model.forward(val_images)

        # Fit temperature
        best_t = self.temp_scaling.fit(logits, val_labels)
        self._calibrated = True

        print(f"Calibration complete: T = {best_t:.4f}")

        # Compute calibration metrics
        cal_probs = self.temp_scaling.calibrate(logits)
        uncal_shifted = logits - np.max(logits, axis=1, keepdims=True)
        uncal_exp = np.exp(uncal_shifted)
        uncal_probs = uncal_exp / np.sum(uncal_exp, axis=1, keepdims=True)

        ece_before = self._compute_ece(uncal_probs, val_labels)
        ece_after = self._compute_ece(cal_probs, val_labels)

        print(f"ECE before calibration: {ece_before:.4f}")
        print(f"ECE after calibration:  {ece_after:.4f}")

        return {
            'temperature': best_t,
            'ece_before': ece_before,
            'ece_after': ece_after,
        }

    def predict_calibrated(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass with calibrated probabilities.

        Args:
            images: (N, 1, 28, 28) or (N, seq_len, 28, 28)

        Returns:
            Same as neural_deduction but with calibrated probabilities
        """
        output = self.neural.neural_deduction(images)

        if self._calibrated:
            output['probabilities'] = self.temp_scaling.calibrate(output['logits'])
            output['confidence'] = np.max(output['probabilities'], axis=-1)
            output['class_ids'] = np.argmax(output['probabilities'], axis=-1)

        return output

    def predict_with_uncertainty(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass with MC Dropout uncertainty estimation.

        Args:
            images: (N, 1, 28, 28)

        Returns:
            Dict with calibrated probabilities + uncertainty measures
        """
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]

        mc_result = self.mc_dropout.predict_with_uncertainty(
            self.neural.model, images
        )

        # Apply temperature scaling to mean probs if calibrated
        if self._calibrated:
            # Recompute from mean logits (approximate)
            mean_logits = np.log(mc_result['mean_probs'] + 1e-10)
            mc_result['calibrated_probs'] = self.temp_scaling.calibrate(mean_logits)
        else:
            mc_result['calibrated_probs'] = mc_result['mean_probs']

        return mc_result

    @staticmethod
    def _compute_ece(probs: np.ndarray, labels: np.ndarray,
                     num_bins: int = 15) -> float:
        """Compute Expected Calibration Error."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        n = len(labels)

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0

        for i in range(num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            count = np.sum(mask)
            if count == 0:
                continue
            bin_acc = np.mean(predictions[mask] == labels[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (count / n) * abs(bin_acc - bin_conf)

        return float(ece)

    def summary(self) -> Dict:
        return {
            'calibrated': self._calibrated,
            'temperature': self.temp_scaling.temperature,
            'mc_dropout_rate': self.mc_dropout.dropout_rate,
            'mc_num_passes': self.mc_dropout.num_passes,
        }
