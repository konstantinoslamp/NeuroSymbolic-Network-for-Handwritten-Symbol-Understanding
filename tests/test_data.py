# test_data.py

import unittest
from src.data.dataset import Dataset
from src.data.loader import DataLoader

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()
        self.dataset.generate_data(num_samples=100)

    def test_data_generation(self):
        self.assertEqual(len(self.dataset.data), 100)
        self.assertTrue(all(isinstance(sample, dict) for sample in self.dataset.data))

    def test_feature_extraction(self):
        features = self.dataset.extract_features()
        self.assertEqual(len(features), 100)
        self.assertTrue(all(isinstance(feature, list) for feature in features))

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()
        self.dataset.generate_data(num_samples=100)
        self.loader = DataLoader(self.dataset, batch_size=10)

    def test_batch_loading(self):
        batches = list(self.loader)
        self.assertEqual(len(batches), 10)
        self.assertEqual(len(batches[0]), 10)

    def test_iterating_over_batches(self):
        for batch in self.loader:
            self.assertEqual(len(batch), 10)

if __name__ == '__main__':
    unittest.main()