"""Eski kod, güvenlik ve kurulum regresyon testleri."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor
from src.model import Activation, NeuralNetwork


class BackpropagationTests(unittest.TestCase):
    def test_numerical_gradient_matches_backpropagation_for_tanh(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(6, 2))
        y = rng.integers(0, 2, size=(6, 1)).astype(float)

        model = NeuralNetwork([2, 3, 1], activation="tanh", learning_rate=1.0)
        model.weights = [w * 0.5 for w in model.weights]
        old_weights = [w.copy() for w in model.weights]
        old_biases = [b.copy() for b in model.biases]

        output = model.forward(X)
        model.backward(X, y, output)
        analytic_w0 = old_weights[0] - model.weights[0]
        analytic_w1 = old_weights[1] - model.weights[1]

        eps = 1e-5

        def loss_with(weights, biases):
            model.weights = weights
            model.biases = biases
            return Activation.binary_cross_entropy(y, model.forward(X))

        numerical_w0 = np.zeros_like(old_weights[0])
        for index in np.ndindex(old_weights[0].shape):
            plus = [w.copy() for w in old_weights]
            minus = [w.copy() for w in old_weights]
            plus[0][index] += eps
            minus[0][index] -= eps
            numerical_w0[index] = (
                loss_with(plus, old_biases) - loss_with(minus, old_biases)
            ) / (2 * eps)

        numerical_w1 = np.zeros_like(old_weights[1])
        for index in np.ndindex(old_weights[1].shape):
            plus = [w.copy() for w in old_weights]
            minus = [w.copy() for w in old_weights]
            plus[1][index] += eps
            minus[1][index] -= eps
            numerical_w1[index] = (
                loss_with(plus, old_biases) - loss_with(minus, old_biases)
            ) / (2 * eps)

        np.testing.assert_allclose(analytic_w0, numerical_w0, atol=2e-5, rtol=1e-4)
        np.testing.assert_allclose(analytic_w1, numerical_w1, atol=2e-5, rtol=1e-4)

    def test_activation_survives_save_and_load(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(4, 2))
        model = NeuralNetwork([2, 3, 1], activation="tanh")
        expected = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "model.json")
            model.save_model(path)
            loaded = NeuralNetwork([2, 3, 1], activation="sigmoid")
            loaded.load_model(path)

        self.assertEqual(loaded.activation_name, "tanh")
        np.testing.assert_allclose(loaded.predict_proba(X), expected)


class DataProcessorTests(unittest.TestCase):
    def test_synthetic_noise_is_reproducible(self):
        processor = DataProcessor()
        X1, y1 = processor.generate_synthetic_data(random_state=123)
        X2, y2 = processor.generate_synthetic_data(random_state=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_digits_binary_target_names_match_encoding(self):
        processor = DataProcessor()
        _, y = processor.load_digits_dataset(binary=True)
        self.assertEqual(processor.target_names[0], "Diğer Rakamlar")
        self.assertEqual(processor.target_names[1], "Rakam 0")
        self.assertEqual(y.min(), 0)
        self.assertEqual(y.max(), 1)


class GitHubPushSafetyTests(unittest.TestCase):
    def test_h100_push_is_opt_in_and_never_force_pushes(self):
        import train_h100

        with patch.object(sys, "argv", ["train_h100.py"]):
            self.assertFalse(train_h100.parse_args().push)
        with patch.object(sys, "argv", ["train_h100.py", "--push"]):
            self.assertTrue(train_h100.parse_args().push)

        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("subprocess.run", side_effect=fake_run):
            train_h100.push_to_github("output_h100")

        flattened = [" ".join(command) for command in calls]
        self.assertTrue(any("git add -- output_h100" in command for command in flattened))
        self.assertFalse(any("-f" in command or "--force" in command for command in flattened))


class DependencyFileTests(unittest.TestCase):
    def test_requirements_do_not_contain_stdlib_or_invalid_syntax(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "requirements.txt"), encoding="utf-8") as handle:
            lines = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
        self.assertNotIn("asyncio", lines)
        self.assertFalse(any(line.startswith("pickle5") for line in lines))
        self.assertFalse(any(line.startswith('"""') for line in lines))

        with open(os.path.join(root, "requirements_web.txt"), encoding="utf-8") as handle:
            web_lines = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
        self.assertFalse(any(line.startswith('"""') for line in web_lines))


if __name__ == "__main__":
    unittest.main()
