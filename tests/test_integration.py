"""
Test the full Neuro-Symbolic Loop
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.integration.training_loop import NeuroSymbolicTrainer
from src.symbolic.symbolic_interface import ArithmeticSymbolicModule
from src.config.task_definition import TASK

class MockNeuralModule:
    """Simulates a neural network for testing"""
    def neural_deduction(self, images):
        batch_size = len(images)
        # Simulate predicting "3 + 6" (indices: 3, 10, 6)
        # Shape: (batch, 3 symbols)
        return {
            'class_ids': np.array([[3, 10, 6]] * batch_size),
            'probabilities': np.random.rand(batch_size, 3, 14)
        }
    
    def neural_induction(self, signals):
        # Simulate training step
        return {'total_loss': 0.5}

def test_training_loop():
    print("🔄 Testing Training Loop...")
    
    # Setup
    neural = MockNeuralModule()
    symbolic = ArithmeticSymbolicModule()
    trainer = NeuroSymbolicTrainer(neural, symbolic, TASK)
    
    # Fake data: 2 images
    images = np.zeros((2, 28, 28))
    # Ground truth: The first image should be 8 (so 3+6 is WRONG -> Abduction needed)
    # The second image should be 9 (so 3+6 is CORRECT -> No abduction)
    ground_truth = [8.0, 9.0]
    
    # Run one step
    metrics = trainer.train_step(images, ground_truth)
    
    print(f"   Metrics: {metrics}")
    
    # Assertions
    assert metrics['correct'] == 1, "Should have 1 correct prediction (the second one)"
    assert metrics['abductions'] == 1, "Should trigger abduction for the first sample (target 8)"
    
    print("✅ Loop logic verified!")

if __name__ == "__main__":
    test_training_loop()