import unittest

import numpy as np

from grid_simulation import optimize_grid
from pso_engine import minimize, project_bounded_sum


class PSOTests(unittest.TestCase):
    def test_projection_respects_bounds_and_sum(self):
        projected = project_bounded_sum(
            np.array([[8.0, -2.0, 3.0], [1.0, 1.0, 1.0]]),
            np.zeros(3),
            np.full(3, 4.0),
            6.0,
        )
        self.assertTrue(np.all(projected >= 0.0))
        self.assertTrue(np.all(projected <= 4.0))
        np.testing.assert_allclose(projected.sum(axis=1), [6.0, 6.0], atol=1e-9)

    def test_pso_minimizes_with_equality_constraint(self):
        result = minimize(
            lambda points: np.sum((points - np.array([1.0, 2.0, 3.0])) ** 2, axis=1),
            np.zeros(3),
            np.full(3, 5.0),
            equality_sum=6.0,
            initial_position=np.array([2.0, 2.0, 2.0]),
            iterations=80,
            seed=12,
        )
        self.assertAlmostEqual(result.position.sum(), 6.0, places=8)
        self.assertLessEqual(result.fitness, result.history[0])


class GridSimulationTests(unittest.TestCase):
    def test_six_bus_dispatch_conserves_total_load(self):
        result = optimize_grid(iterations=80, seed=3)
        np.testing.assert_allclose(result.optimized_loads.sum(), result.baseline_loads.sum())
        self.assertTrue(np.all(result.optimized_loads <= result.bus_capacity + 1e-8))
        self.assertLessEqual(result.optimized_fitness, result.baseline_fitness + 1e-10)
        self.assertEqual(len(result.optimized_line_flows), 5)

    def test_ten_bus_model_and_seed_are_reproducible(self):
        first = optimize_grid(bus_count=10, iterations=60, seed=5)
        second = optimize_grid(bus_count=10, iterations=60, seed=5)
        np.testing.assert_allclose(first.optimized_loads, second.optimized_loads)
        self.assertAlmostEqual(first.optimized_loads.sum(), first.baseline_loads.sum())
        self.assertEqual(len(first.optimized_line_flows), 9)

    def test_invalid_flexible_pool_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "flexible_fraction"):
            optimize_grid(flexible_fraction=0.8)


if __name__ == "__main__":
    unittest.main()
