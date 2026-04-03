"""
Digit Recognizer - Concrete implementation of NeuralModule using our custom CNN
"""

import numpy as np
from typing import Dict, List, Any, Optional
from src.neural.neural_interface import NeuralModule
from src.neural.model import CNN

class DigitRecognizer(NeuralModule):
    def __init__(self):
        self.model = CNN()
        
    def neural_deduction(self, raw_input: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass
        Args:
            raw_input: (batch_size, seq_len, 28, 28) or (batch_size, 28, 28)
        """
        # Handle input shape
        if raw_input.ndim == 3:
            # (batch, 28, 28) -> (batch, 1, 28, 28)
            x = raw_input[:, np.newaxis, :, :]
            batch_size = raw_input.shape[0]
            seq_len = 1
        elif raw_input.ndim == 4:
            # (batch, seq_len, 28, 28) -> (batch * seq_len, 1, 28, 28)
            batch_size, seq_len, h, w = raw_input.shape
            x = raw_input.reshape(batch_size * seq_len, 1, h, w)
        else:
            raise ValueError(f"Unexpected input shape: {raw_input.shape}")
            
        # Run CNN
        logits = self.model.forward(x) # (N, 14)
        
        # Compute probabilities (Softmax)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Get predictions
        class_ids = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        
        # Reshape back to (batch, seq_len, ...)
        if seq_len > 1:
            probs = probs.reshape(batch_size, seq_len, -1)
            logits = logits.reshape(batch_size, seq_len, -1)
            class_ids = class_ids.reshape(batch_size, seq_len)
            confidence = confidence.reshape(batch_size, seq_len)
            
        return {
            'probabilities': probs,
            'logits': logits,
            'class_ids': class_ids,
            'confidence': confidence
        }

    def neural_induction(self, 
                        training_signal: List[Dict],
                        images: np.ndarray,
                        learning_rate: float = 0.001) -> Dict[str, float]:
        """
        Backward pass using training signals from the neuro-symbolic loop
        """
        inputs_to_train = []
        targets_to_train = []
        
        # Mapping symbols to indices (must match Trainer)
        symbol_to_idx = {str(i): i for i in range(10)}
        symbol_to_idx.update({'+': 10, '-': 11, '×': 12, '÷': 13})
        
        for signal in training_signal:
            idx = signal['image_index']
            # images[idx] is (3, 28, 28)
            sample_images = images[idx]
            
            targets = None
            
            if signal['is_correct']:
                # Reinforce current predictions
                targets = [symbol_to_idx[s] for s in signal['original_symbols']]
            elif signal['abductive_targets'] is not None:
                # Use corrected targets
                targets = signal['abductive_targets']
            
            if targets:
                # Add each symbol in the expression to the training batch
                for i in range(len(targets)):
                    # (28, 28) -> (1, 28, 28)
                    img = sample_images[i][np.newaxis, :, :]
                    inputs_to_train.append(img)
                    targets_to_train.append(targets[i])
        
        if not inputs_to_train:
            return {'total_loss': 0.0}
            
        # Stack into batch
        x_batch = np.stack(inputs_to_train) # (N, 1, 28, 28)
        y_batch = np.array(targets_to_train) # (N,)
        
        # Train
        logits = self.model.forward(x_batch)
        loss = self.model.compute_loss(logits, y_batch)
        self.model.backward()
        self.model.update_weights(learning_rate)
        
        return {'total_loss': float(loss)}

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return {
            'conv1_W': self.model.conv1.W,
            'conv1_b': self.model.conv1.b,
            'fc1_W': self.model.fc1.W,
            'fc1_b': self.model.fc1.b,
            'fc2_W': self.model.fc2.W,
            'fc2_b': self.model.fc2.b
        }

    def set_parameters(self, params: Dict[str, np.ndarray]):
        self.model.conv1.W = params['conv1_W']
        self.model.conv1.b = params['conv1_b']
        self.model.fc1.W = params['fc1_W']
        self.model.fc1.b = params['fc1_b']
        self.model.fc2.W = params['fc2_W']
        self.model.fc2.b = params['fc2_b']

    def train_with_gradient(self, images: np.ndarray, gradients: np.ndarray, learning_rate: float = 0.001):
        """
        Train using externally computed gradients (for Semantic Loss)
        Args:
            images: (batch_size, seq_len, 28, 28)
            gradients: (batch_size, seq_len, num_classes) - Gradient of Loss w.r.t Logits
        """
        # Flatten batch and sequence dimensions
        batch_size, seq_len, h, w = images.shape
        num_classes = gradients.shape[-1]
        
        x_flat = images.reshape(batch_size * seq_len, 1, h, w)
        grad_flat = gradients.reshape(batch_size * seq_len, num_classes)
        
        # Forward pass (needed to populate cache for backward)
        self.model.forward(x_flat)
        
        # Backward pass with external gradient
        self.model.backward(grad=grad_flat)
        
        # Update weights
        self.model.update_weights(learning_rate)