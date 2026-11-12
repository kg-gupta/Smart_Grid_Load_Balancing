"""A small, dependency-light Particle Swarm Optimization engine."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


ArrayObjective = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PSOResult:
    position: np.ndarray
    fitness: float
    history: np.ndarray
    evaluations: int


def project_bounded_sum(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_sum: float,
) -> np.ndarray:
    """Project vectors onto box bounds while preserving a required total."""
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if lower.shape != upper.shape or values.shape[-1] != lower.size:
        raise ValueError("values and bounds must have matching final dimensions")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    if target_sum < lower.sum() - 1e-9 or target_sum > upper.sum() + 1e-9:
        raise ValueError("target_sum is infeasible for the supplied bounds")

    low_shift = np.min(values - upper, axis=-1, keepdims=True)
    high_shift = np.max(values - lower, axis=-1, keepdims=True)
    for _ in range(60):
        shift = (low_shift + high_shift) / 2.0
        projected = np.clip(values - shift, lower, upper)
        too_large = projected.sum(axis=-1, keepdims=True) > target_sum
        low_shift = np.where(too_large, shift, low_shift)
        high_shift = np.where(too_large, high_shift, shift)

    projected = np.clip(values - (low_shift + high_shift) / 2.0, lower, upper)
    residual = target_sum - projected.sum(axis=-1)
    for index in range(lower.size):
        room = np.where(residual >= 0, upper[index] - projected[..., index],
                        projected[..., index] - lower[index])
        adjustment = np.sign(residual) * np.minimum(np.abs(residual), room)
        projected[..., index] += adjustment
        residual -= adjustment
    return projected


def minimize(
    objective: ArrayObjective,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    *,
    equality_sum: Optional[float] = None,
    initial_position: Optional[np.ndarray] = None,
    particles: int = 48,
    iterations: int = 250,
    inertia_start: float = 0.9,
    inertia_end: float = 0.4,
    cognitive: float = 1.7,
    social: float = 1.7,
    seed: Optional[int] = None,
) -> PSOResult:
    """Minimize a vectorized objective with box and optional sum constraints."""
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if lower.ndim != 1 or lower.shape != upper.shape or lower.size == 0:
        raise ValueError("bounds must be non-empty one-dimensional arrays")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("bounds must be finite")
    if np.any(lower > upper):
        raise ValueError("lower bounds must not exceed upper bounds")
    if particles < 2 or iterations < 1:
        raise ValueError("particles must be at least 2 and iterations at least 1")
    if equality_sum is not None and (
        equality_sum < lower.sum() - 1e-9 or equality_sum > upper.sum() + 1e-9
    ):
        raise ValueError("equality_sum is infeasible for the supplied bounds")

    rng = np.random.default_rng(seed)
    dimensions = lower.size
    positions = rng.uniform(lower, upper, size=(particles, dimensions))
    if equality_sum is not None:
        positions = project_bounded_sum(positions, lower, upper, equality_sum)
    if initial_position is not None:
        initial = np.asarray(initial_position, dtype=float)
        if initial.shape != (dimensions,) or np.any(~np.isfinite(initial)):
            raise ValueError("initial_position must be a finite vector matching the bounds")
        initial = np.clip(initial, lower, upper)
        if equality_sum is not None:
            initial = project_bounded_sum(initial, lower, upper, equality_sum)
        positions[0] = initial

    velocities = rng.uniform(-1.0, 1.0, size=(particles, dimensions)) * (upper - lower)
    fitness = np.asarray(objective(positions), dtype=float)
    if fitness.shape != (particles,) or np.any(~np.isfinite(fitness)):
        raise ValueError("objective must return one finite value per particle")
    evaluations = particles
    personal_positions = positions.copy()
    personal_fitness = fitness.copy()
    best_index = int(np.argmin(personal_fitness))
    global_position = personal_positions[best_index].copy()
    global_fitness = float(personal_fitness[best_index])
    history = np.empty(iterations + 1, dtype=float)
    history[0] = global_fitness

    for iteration in range(iterations):
        progress = iteration / max(iterations - 1, 1)
        inertia = inertia_start + progress * (inertia_end - inertia_start)
        velocities = (
            inertia * velocities
            + cognitive * rng.random((particles, dimensions))
            * (personal_positions - positions)
            + social * rng.random((particles, dimensions))
            * (global_position - positions)
        )
        previous_positions = positions
        positions = positions + velocities
        positions = np.clip(positions, lower, upper)
        if equality_sum is not None:
            positions = project_bounded_sum(positions, lower, upper, equality_sum)
        velocities = positions - previous_positions

        fitness = np.asarray(objective(positions), dtype=float)
        if fitness.shape != (particles,) or np.any(~np.isfinite(fitness)):
            raise ValueError("objective must return one finite value per particle")
        evaluations += particles
        improved = fitness < personal_fitness
        personal_positions[improved] = positions[improved]
        personal_fitness[improved] = fitness[improved]
        best_index = int(np.argmin(personal_fitness))
        if personal_fitness[best_index] < global_fitness:
            global_position = personal_positions[best_index].copy()
            global_fitness = float(personal_fitness[best_index])
        history[iteration + 1] = global_fitness

    return PSOResult(global_position, global_fitness, history, evaluations)
