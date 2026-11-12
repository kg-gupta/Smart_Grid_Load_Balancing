# Smart Grid Load Balancing with Particle Swarm Optimization

An educational, reproducible simulation that shifts a conserved pool of flexible demand among buses in a radial distribution grid. The included Streamlit dashboard compares the original schedule with the PSO schedule and highlights bus and feeder loading.

## Features

- Pure NumPy particle swarm engine with linearly adaptive inertia, reproducible random seeds, and bounded-sum projection.
- Six- and ten-bus radial grid models with branch-flow calculations.
- Fitness considers bus utilization imbalance, feeder congestion, capacity violations, and unnecessary schedule movement.
- Baseline comparison, convergence history, static plots, and interactive Streamlit charts.
- Focused unit tests for constraint handling and both grid sizes.

## Model and fitness

Each bus starts with a fixed demand and a flexible portion. The optimizer allocates the flexible portion `x_i`, subject to:

```text
0 <= x_i <= bus_capacity_i - fixed_demand_i
sum(x_i) = total_flexible_demand
bus_load_i = fixed_demand_i + x_i
```

For each radial feeder, the estimated real-power flow is the sum of downstream bus loads. The objective minimized is:

```text
J = 10000 * (sum(bus_overload_ratio^2) + sum(feeder_overload_ratio^2))
  + 35 * mean((bus_utilization - mean_bus_utilization)^2)
  +  2 * mean(feeder_utilization^2)
  + 0.05 * mean(((x - baseline_flexible) / total_flexible_demand)^2)
```

The overload terms make capacity violations expensive; the other terms balance utilization, discourage feeder congestion, and avoid needless rescheduling. Bus headroom and feeder capacity can be varied in the demo. These are planning-level approximations, not an AC power-flow solver or a real-time grid controller.

## Run

From this directory:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

Generate and save a static comparison plot:

```powershell
python -c "from grid_simulation import optimize_grid, plot_comparison; plot_comparison(optimize_grid(), 'grid_comparison.png')"
```

## Project layout

```text
smart-grid-pso/
├── pso_engine.py        # Constrained NumPy PSO
├── grid_simulation.py   # Radial grid model, fitness, comparison plots
├── app.py               # Streamlit demo
├── requirements.txt
├── tests/
│   └── test_grid_simulation.py
└── README.md
```
