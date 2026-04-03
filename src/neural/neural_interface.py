"""
Neural Module Interface - Implements neural_deduction and neural_induction
Based on Tsamoura & Michael paper
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np
import pickle

class NeuralModule(ABC):
    """
    Abstract base class for neural components
    
    Provides two key operations:
    1. neural_deduction: Forward pass (perception/prediction)
    2. neural_induction: Backward pass (learning from symbolic feedback)
    """
    
    @abstractmethod
    def neural_deduction(self, raw_input: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass: raw input → predictions
        
        Args:
            raw_input: (batch_size, height, width) images
            
        Returns:
            {
                'predictions': (batch_size, num_classes) probabilities,
                'logits': (batch_size, num_classes) raw scores,
                'confidence': (batch_size,) max probability per sample,
                'class_ids': (batch_size,) argmax predictions
            }
        """
        pass
    
    @abstractmethod
    def neural_induction(self, 
                        training_signal: Dict[str, np.ndarray],
                        learning_rate: float = 0.001) -> Dict[str, float]:
        """
        Backward pass: update parameters using symbolic feedback
        
        Args:
            training_signal: {
                'inputs': (batch_size, height, width),
                'ground_truth': (batch_size,) original labels,
                'abductive_targets': (batch_size, num_classes) soft targets from abduction,
                'abduction_weight': float in [0, 1],
                'use_abduction': (batch_size,) bool mask for which samples to use abduction
            }
            learning_rate: Step size for gradient descent
            
        Returns:
            {
                'classification_loss': float,
                'semantic_loss': float,  # Loss from abductive feedback
                'total_loss': float,
                'num_abductions_used': int
            }
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Return all trainable parameters"""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set trainable parameters"""
        pass
    
    def save_weights(self, filepath: str):
        """Save model weights"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    
    def load_weights(self, filepath: str):
        """Load model weights"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
            self.set_parameters(params)


class EnsembleNeuralModule(NeuralModule):
    """
    Combines multiple specialized neural modules
    Example: DigitRecognizer + OperatorRecognizer
    """
    
    def __init__(self, modules: Dict[str, NeuralModule]):
        """
        Args:
            modules: {
                'digit': DigitRecognizer instance,
                'operator': OperatorRecognizer instance
            }
        """
        self.modules = modules
    
    def neural_deduction(self, raw_input: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run deduction on all modules
        
        Returns:
            {
                'digit': {...predictions from DigitRecognizer...},
                'operator': {...predictions from OperatorRecognizer...}
            }
        """
        results = {}
        for name, module in self.modules.items():
            results[name] = module.neural_deduction(raw_input)
        return results
    
    def neural_induction(self, 
                        training_signal: Dict[str, Dict],
                        learning_rate: float = 0.001) -> Dict[str, Dict[str, float]]:
        """
        Train all modules with their respective signals
        
        Args:
            training_signal: {
                'digit': {...signal for digits...},
                'operator': {...signal for operators...}
            }
        """
        losses = {}
        for name, module in self.modules.items():
            if name in training_signal:
                losses[name] = module.neural_induction(
                    training_signal[name], 
                    learning_rate
                )
        return losses
    
    def get_parameters(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {name: mod.get_parameters() for name, mod in self.modules.items()}
    
    def set_parameters(self, params: Dict[str, Dict[str, np.ndarray]]):
        for name, mod_params in params.items():
            self.modules[name].set_parameters(mod_params)
