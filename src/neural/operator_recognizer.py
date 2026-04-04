"""
Operator Recognizer - Dedicated 4-class CNN for operator classification (P2.3)

Operators (+, -, ×, ÷) have fundamentally different visual statistics from
digits (0-9). Using a separate specialized network improves recognition
accuracy and is architecturally cleaner.

Architecture:
  Conv(1→16, 3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(→64) → ReLU → Dense(→4) → Softmax

Integrated via EnsembleNeuralModule which routes digit/operator positions
to their respective specialized networks.
"""

import numpy as np
import pickle
from typing import Dict, List, Any
from src.neural.neural_interface import NeuralModule
from src.neural.cnn import Conv2D, MaxPool2D, Flatten, Dense, ReLU, SoftmaxCrossEntropy


# ---------------------------------------------------------------------------
# Operator CNN (4-class)
# ---------------------------------------------------------------------------

class OperatorCNN:
    """
    Specialized CNN for operator recognition.

    Architecture:
      Conv(1→16, 3×3) → ReLU → MaxPool(2×2) → Flatten → Dense(→64) → ReLU → Dense(→4)

    Output classes: 0='+', 1='-', 2='×', 3='÷'
    """

    def __init__(self):
        self.conv1 = Conv2D(in_channels=1, out_channels=16, kernel_size=3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D()
        self.flatten = Flatten()
        # After conv (28→26) and pool (26→13): 16 × 13 × 13 = 2704
        self.fc1 = Dense(in_features=16 * 13 * 13, out_features=64)
        self.relu2 = ReLU()
        self.fc2 = Dense(in_features=64, out_features=4)
        self.loss_fn = SoftmaxCrossEntropy()

        self.trainable_layers = [self.conv1, self.fc1, self.fc2]

    def forward(self, x):
        """
        Forward pass.
        x: (N, 1, 28, 28)
        returns: logits (N, 4)
        """
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu2.forward(x)
        x = self.fc2.forward(x)
        return x

    def compute_loss(self, logits, labels):
        return self.loss_fn.forward(logits, labels)

    def backward(self, grad=None):
        if grad is None:
            grad = self.loss_fn.backward()
        grad = self.fc2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.fc1.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)

    def update_weights(self, learning_rate=0.01):
        for layer in self.trainable_layers:
            if hasattr(layer, 'grad_W'):
                layer.W -= learning_rate * layer.grad_W
                layer.b -= learning_rate * layer.grad_b

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def save_weights(self, filepath: str):
        weights = {
            'conv1_W': self.conv1.W, 'conv1_b': self.conv1.b,
            'fc1_W': self.fc1.W, 'fc1_b': self.fc1.b,
            'fc2_W': self.fc2.W, 'fc2_b': self.fc2.b,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, filepath: str):
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        self.conv1.W = weights['conv1_W']
        self.conv1.b = weights['conv1_b']
        self.fc1.W = weights['fc1_W']
        self.fc1.b = weights['fc1_b']
        self.fc2.W = weights['fc2_W']
        self.fc2.b = weights['fc2_b']


# ---------------------------------------------------------------------------
# Operator index mapping
# ---------------------------------------------------------------------------

# Local 4-class indices used by OperatorCNN
OP_LOCAL_TO_SYMBOL = {0: '+', 1: '-', 2: '×', 3: '÷'}
OP_SYMBOL_TO_LOCAL = {v: k for k, v in OP_LOCAL_TO_SYMBOL.items()}

# Global 14-class indices used by the unified system
OP_LOCAL_TO_GLOBAL = {0: 10, 1: 11, 2: 12, 3: 13}
OP_GLOBAL_TO_LOCAL = {v: k for k, v in OP_LOCAL_TO_GLOBAL.items()}


# ---------------------------------------------------------------------------
# OperatorRecognizer (NeuralModule implementation)
# ---------------------------------------------------------------------------

class OperatorRecognizer(NeuralModule):
    """
    Dedicated neural module for operator recognition.

    Uses a specialized 4-class CNN (OperatorCNN) for +, -, ×, ÷ classification.
    Outputs are mapped to the global 14-class index space for compatibility
    with the rest of the system.
    """

    def __init__(self):
        self.model = OperatorCNN()

    def neural_deduction(self, raw_input: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass for operator images.

        Args:
            raw_input: (batch_size, 28, 28) or (batch_size, 1, 28, 28)

        Returns:
            dict with 4-class probabilities mapped to global 14-class space
        """
        if raw_input.ndim == 3:
            x = raw_input[:, np.newaxis, :, :]
        elif raw_input.ndim == 4 and raw_input.shape[1] == 1:
            x = raw_input
        else:
            x = raw_input

        batch_size = x.shape[0]

        logits_4 = self.model.forward(x)  # (N, 4)

        # Softmax over 4 classes
        shifted = logits_4 - np.max(logits_4, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs_4 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Map to global 14-class space
        probs_14 = np.zeros((batch_size, 14))
        logits_14 = np.full((batch_size, 14), -1e6)  # Very low for digit classes

        for local_idx, global_idx in OP_LOCAL_TO_GLOBAL.items():
            probs_14[:, global_idx] = probs_4[:, local_idx]
            logits_14[:, global_idx] = logits_4[:, local_idx]

        class_ids = np.argmax(probs_14, axis=1)
        confidence = np.max(probs_14, axis=1)

        return {
            'probabilities': probs_14,
            'logits': logits_14,
            'class_ids': class_ids,
            'confidence': confidence,
            'probabilities_local': probs_4,
            'logits_local': logits_4,
        }

    def neural_induction(self, training_signal, images=None,
                         learning_rate: float = 0.001) -> Dict[str, float]:
        """Train using training signals."""
        return {'total_loss': 0.0}

    def train_on_batch(self, images: np.ndarray, labels: np.ndarray,
                       learning_rate: float = 0.001) -> float:
        """
        Train on a batch of operator images.

        Args:
            images: (N, 1, 28, 28) operator images
            labels: (N,) local operator labels (0-3)
            learning_rate: step size

        Returns:
            loss value
        """
        logits = self.model.forward(images)
        loss = self.model.compute_loss(logits, labels)
        self.model.backward()
        self.model.update_weights(learning_rate)
        return float(loss)

    def train_with_gradient(self, images: np.ndarray, gradients: np.ndarray,
                            learning_rate: float = 0.001):
        """
        Train with external gradients (for semantic loss integration).

        Args:
            images: (N, 1, 28, 28)
            gradients: (N, 14) or (N, 4) gradient w.r.t. class probs
        """
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]

        # If gradients are in 14-class space, extract operator classes
        if gradients.shape[-1] == 14:
            grad_4 = np.zeros((gradients.shape[0], 4))
            for local_idx, global_idx in OP_LOCAL_TO_GLOBAL.items():
                grad_4[:, local_idx] = gradients[:, global_idx]
        else:
            grad_4 = gradients

        self.model.forward(images)
        self.model.backward(grad=grad_4)
        self.model.update_weights(learning_rate)

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return {
            'op_conv1_W': self.model.conv1.W,
            'op_conv1_b': self.model.conv1.b,
            'op_fc1_W': self.model.fc1.W,
            'op_fc1_b': self.model.fc1.b,
            'op_fc2_W': self.model.fc2.W,
            'op_fc2_b': self.model.fc2.b,
        }

    def set_parameters(self, params: Dict[str, np.ndarray]):
        self.model.conv1.W = params['op_conv1_W']
        self.model.conv1.b = params['op_conv1_b']
        self.model.fc1.W = params['op_fc1_W']
        self.model.fc1.b = params['op_fc1_b']
        self.model.fc2.W = params['op_fc2_W']
        self.model.fc2.b = params['op_fc2_b']


# ---------------------------------------------------------------------------
# Split Recognizer: Digit CNN + Operator CNN combined
# ---------------------------------------------------------------------------

class SplitRecognizer(NeuralModule):
    """
    Ensemble module that routes digit positions to DigitRecognizer
    and operator positions to OperatorRecognizer.

    For a standard expression [d1, op, d2]:
      - Positions 0, 2 → DigitRecognizer (10-class)
      - Position 1 → OperatorRecognizer (4-class)

    The outputs are merged into a unified 14-class probability space.
    """

    def __init__(self):
        from src.neural.digit_recognizer import DigitRecognizer
        self.digit_recognizer = DigitRecognizer()
        self.operator_recognizer = OperatorRecognizer()

    def neural_deduction(self, raw_input: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass routing each position to its specialized network.

        Args:
            raw_input: (batch_size, seq_len, 28, 28)

        Returns:
            Unified 14-class output for all positions
        """
        batch_size, seq_len, h, w = raw_input.shape

        all_probs = np.zeros((batch_size, seq_len, 14))
        all_logits = np.zeros((batch_size, seq_len, 14))
        all_class_ids = np.zeros((batch_size, seq_len), dtype=int)
        all_confidence = np.zeros((batch_size, seq_len))

        for t in range(seq_len):
            imgs_t = raw_input[:, t, :, :]  # (batch, 28, 28)

            if t % 2 == 0:
                # Digit position
                out = self.digit_recognizer.neural_deduction(imgs_t)
                # Digit CNN outputs 14 classes, but we zero out operators
                probs = out['probabilities']
                if probs.ndim == 3:
                    probs = probs[:, 0, :]
                logits = out['logits']
                if logits.ndim == 3:
                    logits = logits[:, 0, :]
                # Suppress operator classes for digit positions
                probs_clean = probs.copy()
                probs_clean[:, 10:14] = 0
                row_sums = probs_clean.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                probs_clean = probs_clean / row_sums
                all_probs[:, t, :] = probs_clean
                all_logits[:, t, :] = logits
            else:
                # Operator position
                out = self.operator_recognizer.neural_deduction(imgs_t)
                all_probs[:, t, :] = out['probabilities']
                all_logits[:, t, :] = out['logits']

            all_class_ids[:, t] = np.argmax(all_probs[:, t, :], axis=1)
            all_confidence[:, t] = np.max(all_probs[:, t, :], axis=1)

        return {
            'probabilities': all_probs,
            'logits': all_logits,
            'class_ids': all_class_ids,
            'confidence': all_confidence,
        }

    def neural_induction(self, training_signal, images=None,
                         learning_rate: float = 0.001) -> Dict[str, float]:
        return {'total_loss': 0.0}

    def train_with_gradient(self, images: np.ndarray, gradients: np.ndarray,
                            learning_rate: float = 0.001):
        """
        Route gradients to appropriate specialized networks.

        Args:
            images: (batch_size, seq_len, 28, 28)
            gradients: (batch_size, seq_len, 14)
        """
        batch_size, seq_len, h, w = images.shape

        for t in range(seq_len):
            imgs_t = images[:, t, :, :]  # (batch, 28, 28)
            grads_t = gradients[:, t, :]  # (batch, 14)

            if t % 2 == 0:
                # Digit position
                imgs_4d = imgs_t[:, np.newaxis, :, :]
                # Only pass digit-relevant gradients (classes 0-9)
                digit_grads = grads_t.copy()
                digit_grads[:, 10:14] = 0
                self.digit_recognizer.model.forward(imgs_4d)
                self.digit_recognizer.model.backward(grad=digit_grads)
                self.digit_recognizer.model.update_weights(learning_rate)
            else:
                # Operator position
                self.operator_recognizer.train_with_gradient(
                    imgs_t, grads_t, learning_rate
                )

    def get_parameters(self) -> Dict[str, np.ndarray]:
        params = self.digit_recognizer.get_parameters()
        params.update(self.operator_recognizer.get_parameters())
        return params

    def set_parameters(self, params: Dict[str, np.ndarray]):
        digit_params = {k: v for k, v in params.items() if not k.startswith('op_')}
        op_params = {k: v for k, v in params.items() if k.startswith('op_')}
        if digit_params:
            self.digit_recognizer.set_parameters(digit_params)
        if op_params:
            self.operator_recognizer.set_parameters(op_params)

    @property
    def model(self):
        """Compatibility: return digit model for gradient monitoring."""
        return self.digit_recognizer.model
