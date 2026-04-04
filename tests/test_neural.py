import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.neural.model import CNN
from src.neural.digit_recognizer import DigitRecognizer


class TestCNN(unittest.TestCase):

    def setUp(self):
        self.model = CNN()
        # (N, 1, 28, 28) — grayscale MNIST-style images
        self.X = np.random.rand(8, 1, 28, 28).astype(np.float32)
        self.y = np.random.randint(0, 14, size=(8,))

    def test_forward_shape(self):
        logits = self.model.forward(self.X)
        self.assertEqual(logits.shape, (8, 14), "Logits should be (batch, 14)")

    def test_compute_loss(self):
        logits = self.model.forward(self.X)
        loss = self.model.compute_loss(logits, self.y)
        self.assertIsInstance(float(loss), float)
        self.assertGreater(float(loss), 0.0)

    def test_weights_change_after_training(self):
        w_before = self.model.conv1.W.copy()
        logits = self.model.forward(self.X)
        self.model.compute_loss(logits, self.y)
        self.model.backward()
        self.model.update_weights(learning_rate=0.01)
        self.assertFalse(np.allclose(w_before, self.model.conv1.W),
                         "Weights should change after a gradient step")

    def test_predict_shape(self):
        preds = self.model.predict(self.X)
        self.assertEqual(preds.shape, (8,))
        self.assertTrue(np.all(preds >= 0) and np.all(preds < 14))

    def test_save_and_load_weights(self):
        path = 'tests/tmp_test_weights.pkl'
        self.model.save_weights(path)
        w_orig = self.model.conv1.W.copy()

        # Corrupt weights, then reload
        self.model.conv1.W[:] = 0.0
        self.model.load_weights(path)
        self.assertTrue(np.allclose(w_orig, self.model.conv1.W),
                        "Loaded weights should match saved weights")
        os.remove(path)


class TestDigitRecognizer(unittest.TestCase):

    def setUp(self):
        self.recognizer = DigitRecognizer()
        # (batch, seq_len, H, W) — 3-symbol expression
        self.images = np.random.rand(4, 3, 28, 28).astype(np.float32)

    def test_neural_deduction_output_keys(self):
        out = self.recognizer.neural_deduction(self.images)
        for key in ('probabilities', 'logits', 'class_ids', 'confidence'):
            self.assertIn(key, out, f"Missing key: {key}")

    def test_neural_deduction_probs_sum_to_one(self):
        out = self.recognizer.neural_deduction(self.images)
        probs = out['probabilities']  # (batch, seq_len, 14)
        sums = probs.sum(axis=-1)
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-5),
                        "Probabilities should sum to 1 along class axis")

    def test_get_set_parameters(self):
        params = self.recognizer.get_parameters()
        self.assertIn('conv1_W', params)
        # Zero out, restore, check
        orig_W = params['conv1_W'].copy()
        params['conv1_W'][:] = 0.0
        self.recognizer.set_parameters(params)
        self.assertTrue(np.all(self.recognizer.model.conv1.W == 0.0))
        params['conv1_W'] = orig_W
        self.recognizer.set_parameters(params)


if __name__ == '__main__':
    unittest.main()
