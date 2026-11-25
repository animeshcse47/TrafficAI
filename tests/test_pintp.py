import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pintp.data import corrupt_labels, graph_adr, telemetry_data
from pintp.models import GraphMLP, TelemetryMLP
from pintp.training import fit_graph, fit_telemetry


class PINTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_counts(self):
        for model, expected in (
            (GraphMLP(100, False), 17921),
            (GraphMLP(100, True), 17924),
            (TelemetryMLP(False), 12929),
            (TelemetryMLP(True), 12932),
        ):
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_adr_solution_and_clean_holdout(self):
        graph = graph_adr(42)
        self.assertEqual(graph.values.shape, (300, 100))
        self.assertEqual(graph.degree.sum(), 964)
        np.testing.assert_allclose(graph.laplacian.sum(axis=1), 0)
        self.assertLess(graph.times[graph.split - 1], graph.times[graph.split])
        time = graph.times[210:-10, None]
        velocity = 0.5 * (1 + 0.3 * np.sin(np.pi * time / 5) + 0.2 * np.sin(np.pi * time))
        diffusion = 0.1 / (0.3 + 0.7 * velocity / 0.5)
        source = (1 + 0.3 * np.sin(2 * np.pi * time / 5) + 0.2 * np.cos(np.pi * time)) * graph.source
        derivative = np.gradient(graph.values[200:], graph.times[200:], axis=0)[10:-10]
        residual = derivative + (velocity + diffusion) * (graph.values[210:-10] @ graph.laplacian) - source
        self.assertLess(np.mean(residual ** 2), 1e-7)
        original = graph.values.copy()
        noisy = corrupt_labels(graph.values[:graph.split], "high", 42)
        self.assertFalse(np.array_equal(noisy, original[:graph.split]))
        np.testing.assert_array_equal(graph.values, original)

    def test_real_csv_path_uses_train_only_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "fixture.csv"
            t = np.arange(400)
            t = np.r_[t[::-1], 101]
            frame = pd.DataFrame({
                "id": ["m_2274"] * 400 + ["other"],
                "t": t, "cpu": t, "mem": t, "pg": 0, "ki": 0,
                "net_in": t, "net_out": t, "disk": t,
            })
            frame.to_csv(csv, index=False, header=False)
            data = telemetry_data(csv)
            self.assertEqual(data.source, "alibaba_v2018")
            self.assertEqual(data.split, 320)
            np.testing.assert_allclose(data.features[data.split - 1], 1)
            self.assertGreater(data.values[-1], 1)
            self.assertGreater(data.features[-1, 0], 1)
            with self.assertRaisesRegex(ValueError, "was not found"):
                telemetry_data(csv, "missing")

    def test_both_physics_models_train(self):
        graph = graph_adr(42)
        _, prediction, residual = fit_graph(graph, graph.values[:graph.split], 0.03, 1, 42, "cpu")
        self.assertEqual(prediction.shape, (100, 100))
        self.assertTrue(np.isfinite(prediction).all() and np.isfinite(residual))
        telemetry = telemetry_data(None)
        _, prediction, residual = fit_telemetry(telemetry, np.arange(287), 1, 2, 42, "cpu")
        self.assertEqual(prediction.shape, (1455,))
        self.assertTrue(np.isfinite(prediction).all() and np.isfinite(residual))

    def test_ode_derivative_uses_time_only(self):
        model = TelemetryMLP(True)
        model.predictor = Quadratic()
        with torch.no_grad():
            model.coefficients.copy_(torch.tensor([0.5, 0.1, 0.2]))
        features = torch.tensor([[0.0, 3.0, 0, 0, 0], [2.0, 8.0, 0, 0, 0]])
        expected = 2 - features[:, 0] - 0.1 * (features[:, 0] ** 2 + features[:, 1]) - 0.2
        torch.testing.assert_close(model.residual(features), expected)


class Quadratic(torch.nn.Module):
    def forward(self, features):
        return features[:, :1].square() + features[:, 1:2]


if __name__ == "__main__":
    unittest.main()
