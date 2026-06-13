"""
SwarmBench Evaluation Framework
================================
Enterprise-grade AI agent benchmarking harness.
Evaluates multi-agent analytical reasoning, code safety, and cross-source reconciliation.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json

st.set_page_config(
    page_title="SwarmBench | AI Evaluation Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialization ──────────────────────────────────────────────
if "session_log" not in st.session_state:
    st.session_state.session_log = []
if "scores" not in st.session_state:
    st.session_state.scores = {}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #8b949e !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }
    .stMetric [data-testid="stMetricValue"] { color: #58a6ff !important; font-family: 'JetBrains Mono', monospace; }
    .benchmark-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; font-family: monospace; }
    .badge-pass { background: #1a3a2a; color: #3fb950; border: 1px solid #3fb950; }
    .badge-fail { background: #3a1a1a; color: #f85149; border: 1px solid #f85149; }
    .badge-warn { background: #3a2a10; color: #d29922; border: 1px solid #d29922; }
    .module-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px; margin-bottom: 12px; }
    .code-block { font-family: 'JetBrains Mono', 'Courier New', monospace; font-size: 13px; }
    div[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SwarmBench")
    st.caption("AI Agent Evaluation Framework v2.0")
    st.divider()

    module = st.radio(
        "Select Benchmark Module",
        options=[
            "🏠 Dashboard",
            "🧬 Module 1 — Distribution Recovery",
            "🛡️ Module 2 — AST Code Safety",
            "🔄 Module 3 — Log Reconciliation",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Session Log**")
    if st.session_state.session_log:
        for entry in st.session_state.session_log[-5:]:
            st.caption(f"✓ {entry}")
    else:
        st.caption("No benchmarks run yet.")

    st.divider()
    if st.button("⬇ Export Session Report", use_container_width=True):
        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "scores": st.session_state.scores,
            "log": st.session_state.session_log,
        }
        st.download_button(
            "Download JSON",
            data=json.dumps(report, indent=2),
            file_name=f"swarm_bench_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )


# ── Dashboard ──────────────────────────────────────────────────────────────────
if module == "🏠 Dashboard":
    st.title("SwarmBench Evaluation Framework")
    st.markdown(
        "**Enterprise-grade harness for evaluating AI agents on analytical reasoning, "
        "code safety verification, and multi-source data reconciliation.**"
    )
    st.divider()

    col1, col2, col3 = st.columns(3)
    scores = st.session_state.scores

    with col1:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown("#### 🧬 Distribution Recovery")
        st.caption("Tests statistical self-healing under noise injection")
        ks = scores.get("module1_ks", "—")
        pv = scores.get("module1_pval", "—")
        st.metric("KS Statistic", f"{ks:.4f}" if isinstance(ks, float) else ks)
        st.metric("p-value", f"{pv:.4f}" if isinstance(pv, float) else pv)
        if isinstance(pv, float):
            badge = "PASS" if pv > 0.05 else "FAIL"
            css = "badge-pass" if pv > 0.05 else "badge-fail"
            st.markdown(f'<span class="benchmark-badge {css}">{badge}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown("#### 🛡️ AST Code Safety")
        st.caption("Static analysis + sandboxed execution scoring")
        passed = scores.get("module2_passed", "—")
        total = scores.get("module2_total", "—")
        st.metric("Tests Passed", f"{passed}/{total}" if isinstance(passed, int) else passed)
        if isinstance(passed, int) and isinstance(total, int):
            badge = "PASS" if passed == total else "PARTIAL"
            css = "badge-pass" if passed == total else "badge-warn"
            st.markdown(f'<span class="benchmark-badge {css}">{badge}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="module-card">', unsafe_allow_html=True)
        st.markdown("#### 🔄 Log Reconciliation")
        st.caption("Cross-source anomaly detection accuracy")
        f1 = scores.get("module3_f1", "—")
        st.metric("F1-Score", f"{f1:.4f}" if isinstance(f1, float) else f1)
        if isinstance(f1, float):
            badge = "PERFECT" if f1 == 1.0 else ("PASS" if f1 >= 0.8 else "FAIL")
            css = "badge-pass" if f1 >= 0.8 else "badge-fail"
            st.markdown(f'<span class="benchmark-badge {css}">{badge}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.info("👈 Select a module from the sidebar to run an evaluation.")
    st.markdown("""
    | Module | What it Evaluates | Core Technique |
    |---|---|---|
    | Distribution Recovery | Can an agent restore a corrupted distribution? | KS-Test, Wasserstein Distance |
    | AST Code Safety | Is AI-generated code safe to execute? | Abstract Syntax Tree static analysis |
    | Log Reconciliation | Can an agent detect silent data drops? | Precision / Recall / F1 across sources |
    """)


# ── Module 1 ───────────────────────────────────────────────────────────────────
elif module == "🧬 Module 1 — Distribution Recovery":
    from pages.module1_distribution import render
    render()

# ── Module 2 ───────────────────────────────────────────────────────────────────
elif module == "🛡️ Module 2 — AST Code Safety":
    from pages.module2_ast import render
    render()

# ── Module 3 ───────────────────────────────────────────────────────────────────
elif module == "🔄 Module 3 — Log Reconciliation":
    from pages.module3_reconciliation import render
    render()
