"""Radial smart-grid load model, PSO dispatch, and comparison plots."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from pso_engine import PSOResult, minimize


BUS_NAMES_6 = tuple("Bus {}".format(index) for index in range(1, 7))
BASE_LOADS_6 = np.array([80.0, 70.0, 50.0, 60.0, 55.0, 45.0])
PARENTS_6 = (-1, 0, 1, 1, 3, 3)
LINE_RATINGS_6 = np.array([400.0, 90.0, 175.0, 65.0, 55.0])

BUS_NAMES_10 = tuple("Bus {}".format(index) for index in range(1, 11))
BASE_LOADS_10 = np.array([80.0, 70.0, 50.0, 60.0, 55.0, 45.0, 65.0, 40.0, 50.0, 35.0])
PARENTS_10 = (-1, 0, 1, 1, 3, 3, 0, 6, 6, 8)
LINE_RATINGS_10 = np.array([610.0, 90.0, 175.0, 65.0, 55.0, 170.0, 55.0, 100.0, 45.0])


@dataclass(frozen=True)
class GridResult:
    bus_names: tuple
    baseline_loads: np.ndarray
    optimized_loads: np.ndarray
    baseline_flexible: np.ndarray
    optimized_flexible: np.ndarray
    bus_capacity: np.ndarray
    baseline_line_flows: np.ndarray
    optimized_line_flows: np.ndarray
    line_capacity: np.ndarray
    baseline_fitness: float
    optimized_fitness: float
    history: np.ndarray
    evaluations: int


def _grid_data(bus_count: int):
    if bus_count == 6:
        return BUS_NAMES_6, BASE_LOADS_6, PARENTS_6, LINE_RATINGS_6
    if bus_count == 10:
        return BUS_NAMES_10, BASE_LOADS_10, PARENTS_10, LINE_RATINGS_10
    raise ValueError("bus_count must be 6 or 10")


def _line_flows(loads: np.ndarray, parents: tuple) -> np.ndarray:
    subtree_loads = np.asarray(loads, dtype=float).copy()
    for bus in range(len(parents) - 1, 0, -1):
        subtree_loads[parents[bus]] += subtree_loads[bus]
    return np.array([subtree_loads[bus] for bus in range(1, len(parents))])


def _fitness(
    flexible: np.ndarray,
    fixed_loads: np.ndarray,
    baseline_flexible: np.ndarray,
    bus_capacity: np.ndarray,
    line_capacity: np.ndarray,
    parents: tuple,
) -> np.ndarray:
    loads = fixed_loads + flexible
    bus_utilization = loads / bus_capacity
    line_flows = np.stack([_line_flows(row, parents) for row in loads])
    line_utilization = line_flows / line_capacity
    bus_overload = np.maximum(loads - bus_capacity, 0.0) / bus_capacity
    line_overload = np.maximum(line_flows - line_capacity, 0.0) / line_capacity
    imbalance = np.mean(
        (bus_utilization - np.mean(bus_utilization, axis=1, keepdims=True)) ** 2,
        axis=1,
    )
    line_congestion = np.mean(line_utilization ** 2, axis=1)
    overload_penalty = 10000.0 * (
        np.sum(bus_overload ** 2, axis=1) + np.sum(line_overload ** 2, axis=1)
    )
    movement_penalty = 0.05 * np.mean(
        ((flexible - baseline_flexible) / max(float(baseline_flexible.sum()), 1e-9)) ** 2,
        axis=1,
    )
    return overload_penalty + 35.0 * imbalance + 2.0 * line_congestion + movement_penalty


def optimize_grid(
    base_loads: Optional[np.ndarray] = None,
    *,
    bus_count: int = 6,
    flexible_fraction: float = 0.2,
    line_capacity_factor: float = 1.0,
    bus_headroom: float = 0.35,
    particles: int = 48,
    iterations: int = 250,
    seed: int = 7,
) -> GridResult:
    """Redistribute a conserved flexible-load pool across a 6- or 10-bus grid."""
    names, default_loads, parents, nominal_line_capacity = _grid_data(bus_count)
    loads = np.asarray(default_loads if base_loads is None else base_loads, dtype=float)
    if loads.shape != default_loads.shape or np.any(~np.isfinite(loads)) or np.any(loads <= 0):
        raise ValueError("base_loads must contain one positive finite load per bus")
    if not 0.0 <= flexible_fraction <= 0.5:
        raise ValueError("flexible_fraction must be between 0 and 0.5")
    if line_capacity_factor <= 0 or bus_headroom < 0:
        raise ValueError("line_capacity_factor must be positive and bus_headroom non-negative")

    baseline_flexible = loads * flexible_fraction
    total_flexible = float(baseline_flexible.sum())
    fixed_loads = loads - baseline_flexible
    bus_capacity = loads * (1.0 + bus_headroom)
    line_capacity = nominal_line_capacity * line_capacity_factor
    lower = np.zeros_like(loads)
    upper = np.maximum(bus_capacity - fixed_loads, 0.0)
    if upper.sum() < total_flexible - 1e-9:
        raise ValueError("bus capacities are too low to place the flexible load")

    def objective(population: np.ndarray) -> np.ndarray:
        return _fitness(
            population, fixed_loads, baseline_flexible, bus_capacity, line_capacity, parents
        )

    baseline_fitness = float(objective(baseline_flexible[np.newaxis, :])[0])
    pso: PSOResult = minimize(
        objective,
        lower,
        upper,
        equality_sum=total_flexible,
        initial_position=baseline_flexible,
        particles=particles,
        iterations=iterations,
        seed=seed,
    )
    optimized_loads = fixed_loads + pso.position
    return GridResult(
        bus_names=names,
        baseline_loads=loads.copy(),
        optimized_loads=optimized_loads,
        baseline_flexible=baseline_flexible,
        optimized_flexible=pso.position,
        bus_capacity=bus_capacity,
        baseline_line_flows=_line_flows(loads, parents),
        optimized_line_flows=_line_flows(optimized_loads, parents),
        line_capacity=line_capacity,
        baseline_fitness=baseline_fitness,
        optimized_fitness=pso.fitness,
        history=pso.history,
        evaluations=pso.evaluations,
    )


def plot_comparison(result: GridResult, save_path: Optional[str] = None):
    """Plot baseline and optimized bus loads and feeder utilization."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    positions = np.arange(len(result.bus_names))
    axes[0].bar(positions - 0.2, result.baseline_loads, width=0.4, label="Baseline")
    axes[0].bar(positions + 0.2, result.optimized_loads, width=0.4, label="PSO optimized")
    axes[0].plot(positions, result.bus_capacity, "k--", label="Bus capacity")
    axes[0].set_xticks(positions, result.bus_names, rotation=35, ha="right")
    axes[0].set_ylabel("Load (MW)")
    axes[0].set_title("Bus load allocation")
    axes[0].legend()

    line_positions = np.arange(len(result.line_capacity))
    baseline_util = result.baseline_line_flows / result.line_capacity * 100.0
    optimized_util = result.optimized_line_flows / result.line_capacity * 100.0
    axes[1].bar(line_positions - 0.2, baseline_util, width=0.4, label="Baseline")
    axes[1].bar(line_positions + 0.2, optimized_util, width=0.4, label="PSO optimized")
    axes[1].axhline(100.0, color="red", linestyle="--", label="Thermal limit")
    axes[1].set_xticks(line_positions, ["{}-{}".format(i, j) for i, j in _edge_labels(
        _grid_data(len(result.bus_names))[2]
    )])
    axes[1].set_ylabel("Feeder utilization (%)")
    axes[1].set_title("Feeder loading")
    axes[1].legend()
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(Path(save_path), dpi=150, bbox_inches="tight")
    return figure


def _edge_labels(parents: tuple):
    return [(parent + 1, bus + 1) for bus, parent in enumerate(parents) if bus > 0]
