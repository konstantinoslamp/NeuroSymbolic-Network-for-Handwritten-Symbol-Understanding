import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.helpers import log_message, validate_input, save_model, load_model


class TestHelpers(unittest.TestCase):

    def test_log_message_runs(self):
        # Should not raise
        log_message("test message")

    def test_validate_input_correct_type(self):
        self.assertTrue(validate_input("hello", str))
        self.assertTrue(validate_input(42, int))
        self.assertTrue(validate_input([1, 2], list))

    def test_validate_input_wrong_type(self):
        self.assertFalse(validate_input("hello", int))
        self.assertFalse(validate_input(42, str))

    def test_save_and_load_model(self):
        path = 'tests/tmp_test_model.pkl'
        obj = {'weights': [1.0, 2.0, 3.0], 'bias': 0.5}
        save_model(obj, path)
        self.assertTrue(os.path.exists(path))
        loaded = load_model(path)
        self.assertEqual(loaded['weights'], obj['weights'])
        self.assertEqual(loaded['bias'], obj['bias'])
        os.remove(path)


if __name__ == '__main__':
    unittest.main()
