import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import Dataset
from src.data.loader import DataLoader


class TestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset(num_samples=100, num_features=10)

    def test_data_shape(self):
        data = self.dataset.get_data()
        self.assertEqual(data.shape, (100, 10))

    def test_data_values_in_range(self):
        data = self.dataset.get_data()
        self.assertTrue(np.all(data >= 0.0) and np.all(data <= 1.0))

    def test_generate_data_returns_array(self):
        data = self.dataset.generate_data()
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (100, 10))


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.dataset = Dataset(num_samples=100, num_features=10)
        self.loader = DataLoader(self.dataset.get_data(), batch_size=10)

    def test_batch_count(self):
        batches = list(self.loader)
        self.assertEqual(len(batches), 10)

    def test_batch_size(self):
        batches = list(self.loader)
        for batch in batches:
            self.assertEqual(len(batch), 10)

    def test_reset(self):
        list(self.loader)           # exhaust iterator
        self.loader.reset()
        batches = list(self.loader)
        self.assertEqual(len(batches), 10)


if __name__ == '__main__':
    unittest.main()
