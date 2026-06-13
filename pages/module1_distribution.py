"""
Module 1: Distribution Mutation & Recovery Test
------------------------------------------------
Injects configurable noise into a baseline distribution and scores
an agent's recovery using non-parametric statistical tests.
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def render():
    st.title("🧬 Distribution Mutation & Statistical Recovery Harness")
    st.markdown(
        "Injects **non-linear noise** and **missingness** into a baseline distribution, "
        "then evaluates recovery quality using KS-Test and Wasserstein Distance."
    )
    st.divider()

    # ── Configuration Panel ──────────────────────────────────────────────────
    with st.expander("⚙️ Benchmark Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_size = st.slider("Sample Size", 100, 5000, 500, step=100)
            random_seed = st.number_input("Random Seed", value=42, step=1)
        with col2:
            corruption_rate = st.slider("Corruption Rate (%)", 1, 40, 15) / 100
            null_rate = st.slider("Null Injection Rate (%)", 1, 20, 10) / 100
        with col3:
            dist_type = st.selectbox("Baseline Distribution", ["Lognormal", "Normal", "Gamma", "Pareto"])
            noise_multiplier = st.slider("Noise Multiplier Range", 1.5, 10.0, 5.0)

    # ── Data Generation ──────────────────────────────────────────────────────
    rng = np.random.default_rng(int(random_seed))

    if dist_type == "Lognormal":
        baseline = rng.lognormal(mean=3.0, sigma=0.4, size=sample_size)
    elif dist_type == "Normal":
        baseline = rng.normal(loc=50, scale=10, size=sample_size)
    elif dist_type == "Gamma":
        baseline = rng.gamma(shape=2.0, scale=10.0, size=sample_size)
    else:  # Pareto
        baseline = rng.pareto(a=2.0, size=sample_size) * 50

    # Mutation
    corruption_mask = rng.random(sample_size) < corruption_rate
    null_mask = rng.random(sample_size) < null_rate
    mutated = baseline.copy()
    mutated[corruption_mask] *= rng.uniform(2.5, noise_multiplier, size=sum(corruption_mask))
    mutated[null_mask] = np.nan

    # ── Recovery Strategy ────────────────────────────────────────────────────
    strategy = st.selectbox(
        "Simulated Agent Recovery Strategy",
        ["IQR Clipping + Median Imputation", "Z-Score Clipping + Mean Imputation", "Winsorization (5/95)"]
    )

    cleaned = mutated.copy()

    if strategy == "IQR Clipping + Median Imputation":
        median_val = np.nanmedian(baseline)
        cleaned[np.isnan(cleaned)] = median_val
        q25, q75 = np.nanpercentile(cleaned, [25, 75])
        iqr = q75 - q25
        upper = q75 + 1.5 * iqr
        lower = q25 - 1.5 * iqr
        cleaned = np.clip(cleaned, lower, upper)

    elif strategy == "Z-Score Clipping + Mean Imputation":
        mean_val = np.nanmean(baseline)
        cleaned[np.isnan(cleaned)] = mean_val
        z = np.abs(stats.zscore(cleaned))
        cleaned[z > 3] = mean_val

    else:  # Winsorization
        cleaned[np.isnan(cleaned)] = np.nanmedian(baseline)
        p5, p95 = np.nanpercentile(cleaned, [5, 95])
        cleaned = np.clip(cleaned, p5, p95)

    # ── Evaluation ───────────────────────────────────────────────────────────
    ks_stat, p_value = stats.ks_2samp(baseline, cleaned)
    wasserstein_dist = stats.wasserstein_distance(baseline, cleaned)
    mean_drift = abs(np.mean(baseline) - np.mean(cleaned)) / np.mean(baseline) * 100
    std_drift = abs(np.std(baseline) - np.std(cleaned)) / np.std(baseline) * 100

    benchmark_passed = p_value > 0.05

    # ── Results Layout ────────────────────────────────────────────────────────
    st.divider()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("KS Statistic ↓", f"{ks_stat:.4f}", delta=f"{'✓ Low' if ks_stat < 0.05 else '✗ High'}")
    col_b.metric("p-value ↑", f"{p_value:.4f}", delta="target > 0.05")
    col_c.metric("Mean Drift", f"{mean_drift:.2f}%")
    col_d.metric("Std Drift", f"{std_drift:.2f}%")

    if benchmark_passed:
        st.success("✅ BENCHMARK PASSED — Distribution restored within acceptable tolerance. Null hypothesis retained (p > 0.05).")
    else:
        st.error("❌ BENCHMARK FAILED — Cleaning introduced structural bias. Distributions are statistically dissimilar.")

    # ── Visual Comparison ─────────────────────────────────────────────────────
    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Distribution Comparison**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(baseline, bins=50, alpha=0.6, color="#58a6ff", label="Baseline", density=True)
        ax.hist(cleaned, bins=50, alpha=0.6, color="#3fb950", label="Recovered", density=True)
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")
        ax.legend(facecolor="#0d1117", labelcolor="white")
        ax.set_xlabel("Value", color="#8b949e")
        ax.set_ylabel("Density", color="#8b949e")
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("**Corruption Summary Table**")
        summary = pd.DataFrame({
            "Metric": ["Total Samples", "Corrupted Values", "Null Injections", "Net Cleaned"],
            "Count": [
                sample_size,
                int(sum(corruption_mask)),
                int(sum(null_mask)),
                int(sum(corruption_mask)) + int(sum(null_mask)),
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("**Wasserstein Distance**")
        st.info(f"Earth Mover's Distance: **{wasserstein_dist:.4f}** — measures the work needed to transform recovered → baseline distribution.")

    # ── Log to session ────────────────────────────────────────────────────────
    score_label = "PASS" if benchmark_passed else "FAIL"
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] Module 1 ({dist_type}, n={sample_size}) → KS={ks_stat:.4f}, p={p_value:.4f} [{score_label}]"
    if log_entry not in st.session_state.session_log:
        st.session_state.session_log.append(log_entry)
    st.session_state.scores["module1_ks"] = ks_stat
    st.session_state.scores["module1_pval"] = p_value
