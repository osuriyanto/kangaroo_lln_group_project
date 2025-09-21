import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kangaroo Means", layout="centered")
st.title("Kangaroo Mean Explorer")

# Sidebar controls
num_samples = st.sidebar.slider("Samples per trial", 25, 100, 50)
iterations  = st.sidebar.slider("Number of trials", 1_000, 100_000, 10_000, step=1_000)
seed        = st.sidebar.number_input("Random seed", 0, value=42, step=1)
show_cum    = st.sidebar.checkbox("Show cumulative mean (LLN view)", value=True)

@st.cache_data
def build_population(seed_val: int) -> np.ndarray:
    rng = np.random.default_rng(seed_val)
    return rng.normal(loc=8.0, scale=1.0, size=10_000)

population = build_population(seed)
pop_mean = float(population.mean())

# Population histogram
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.hist(population, bins=30, alpha=0.6)
ax1.axvline(pop_mean, color="red", linestyle="--", label=f"True population mean = {pop_mean:.2f}")
ax1.legend()
st.pyplot(fig1)

@st.cache_data
def simulate_means(pop: np.ndarray, trials: int, n: int, seed_val: int):
    rng = np.random.default_rng(seed_val + 1)
    idx = rng.integers(0, len(pop), size=(trials, n))
    sample_means = pop[idx].mean(axis=1)
    cum_means = np.cumsum(sample_means) / np.arange(1, trials + 1)
    return sample_means, cum_means

sample_means, cum_means = simulate_means(population, iterations, num_samples, seed)
y = cum_means if show_cum else sample_means
label = "Cumulative mean" if show_cum else "Mean of each trial"

# Line plot
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(np.arange(1, iterations + 1), y, label=label)
ax2.axhline(pop_mean, color="red", linestyle="--", label=f"True population mean = {pop_mean:.2f}")
ax2.set_title(f"Sample size = {num_samples}, trials = {iterations}")
ax2.set_xlabel("Trial")
ax2.set_ylabel("Average age of kangaroos")
ax2.legend()
st.pyplot(fig2)