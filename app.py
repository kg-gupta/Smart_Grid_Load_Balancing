"""Streamlit dashboard for the smart-grid PSO load-balancing demo."""
import pandas as pd
import numpy as np
import streamlit as st

from grid_simulation import (
    BASE_LOADS_6,
    BASE_LOADS_10,
    optimize_grid,
)


st.set_page_config(page_title="Smart Grid Load Balancing", page_icon="⚡", layout="wide")
st.title("⚡ Smart Grid Load Balancing")
st.caption("Particle Swarm Optimization redistributes flexible demand to reduce congestion.")

with st.sidebar:
    st.header("Scenario")
    bus_count = st.selectbox("Grid model", [6, 10], index=0, format_func=lambda value: "{}-bus radial grid".format(value))
    defaults = BASE_LOADS_6 if bus_count == 6 else BASE_LOADS_10
    with st.expander("Bus demand (MW)", expanded=False):
        loads = np.array([
            st.slider("Bus {} demand".format(index + 1), 10.0, 150.0, float(load), 1.0, key="load_{}_{}".format(bus_count, index))
            for index, load in enumerate(defaults)
        ])
    flexible_percent = st.slider("Flexible demand (%)", 0, 40, 20, 1)
    line_capacity_factor = st.slider("Feeder capacity factor", 0.60, 1.40, 1.0, 0.05)
    bus_headroom_percent = st.slider("Bus capacity headroom (%)", 0, 60, 35, 1)
    iterations = st.slider("PSO iterations", 50, 500, 250, 50)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=7, step=1)

try:
    result = optimize_grid(
        loads,
        bus_count=bus_count,
        flexible_fraction=flexible_percent / 100.0,
        line_capacity_factor=line_capacity_factor,
        bus_headroom=bus_headroom_percent / 100.0,
        iterations=iterations,
        seed=int(seed),
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

baseline_overload = np.maximum(result.baseline_line_flows - result.line_capacity, 0.0).sum()
optimized_overload = np.maximum(result.optimized_line_flows - result.line_capacity, 0.0).sum()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Flexible demand", "{:.1f} MW".format(result.baseline_flexible.sum()))
col2.metric("Baseline feeder overload", "{:.1f} MW".format(baseline_overload))
col3.metric("Optimized feeder overload", "{:.1f} MW".format(optimized_overload), delta="{:.1f} MW".format(optimized_overload - baseline_overload), delta_color="inverse")
col4.metric("Objective improvement", "{:.1f}%".format(
    100.0 * (result.baseline_fitness - result.optimized_fitness) / max(abs(result.baseline_fitness), 1e-9)
))

st.subheader("Bus load allocation")
chart_data = pd.DataFrame({
    "Bus": result.bus_names,
    "Baseline (MW)": result.baseline_loads,
    "PSO optimized (MW)": result.optimized_loads,
    "Bus capacity (MW)": result.bus_capacity
})

st.bar_chart(
    chart_data,
    x="Bus",
    y=["Baseline (MW)", "PSO optimized (MW)", "Bus capacity (MW)"],
    x_label="Bus",
    y_label="Load / Capacity (MW)"
)

st.subheader("Feeder utilization")
line_labels = ["{}-{}".format(parent + 1, bus + 1)
               for bus, parent in enumerate(
                   ((-1, 0, 1, 1, 3, 3) if bus_count == 6
                    else (-1, 0, 1, 1, 3, 3, 0, 6, 6, 8))
               ) if bus > 0]
baseline_utilization = result.baseline_line_flows / result.line_capacity * 100.0
optimized_utilization = result.optimized_line_flows / result.line_capacity * 100.0
st.bar_chart({
    "Baseline (%)": baseline_utilization,
    "PSO optimized (%)": optimized_utilization,
    "Thermal limit (%)": np.full(len(line_labels), 100.0),
}, x=line_labels)

left, right = st.columns(2)
with left:
    st.subheader("Flexible-load schedule")
    st.dataframe({
        "Bus": result.bus_names,
        "Before (MW)": np.round(result.baseline_flexible, 2),
        "After (MW)": np.round(result.optimized_flexible, 2),
        "Change (MW)": np.round(result.optimized_flexible - result.baseline_flexible, 2),
    }, hide_index=True, use_container_width=True)
with right:
    st.subheader("Convergence")
    st.line_chart({"Best fitness": result.history})

st.info(
    "This is an educational radial-grid approximation. It conserves total demand but "
    "does not run AC power flow, model voltage/reactive power, or send control commands "
    "to real equipment."
)
