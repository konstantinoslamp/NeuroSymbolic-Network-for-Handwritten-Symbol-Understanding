import unittest
import numpy as np
from src.neural.model import NeuralModel

class TestNeuralModel(unittest.TestCase):

    def setUp(self):
        self.model = NeuralModel(input_size=10, hidden_size=5, output_size=2)
        self.X = np.random.rand(100, 10)  # 100 samples, 10 features
        self.y = np.random.randint(0, 2, size=(100, 2))  # 100 samples, 2 classes (one-hot)

    def test_forward_pass(self):
        output = self.model.forward(self.X)
        self.assertEqual(output.shape, (100, 2), "Output shape should match the number of samples and output size.")

    def test_training(self):
        initial_weights = self.model.weights.copy()
        self.model.train(self.X, self.y, epochs=10, learning_rate=0.01)
        self.assertFalse(np.array_equal(initial_weights, self.model.weights), "Weights should change after training.")

    def test_prediction(self):
        self.model.train(self.X, self.y, epochs=10, learning_rate=0.01)
        predictions = self.model.predict(self.X)
        self.assertEqual(predictions.shape, (100, 2), "Predictions shape should match the number of samples and output size.")

if __name__ == '__main__':
    unittest.main()