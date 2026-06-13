"""
Module 3: Multi-Source Log Reconciliation Engine
-------------------------------------------------
Cross-references structured CSV ledger data against async JSON server logs
to surface silent discrepancies. Scores detection accuracy via Precision/Recall/F1.
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime


# ── Ground Truth Datasets ─────────────────────────────────────────────────────
DEFAULT_LEDGER = [
    {"transaction_id": "TX_101", "amount": 150.00,  "status": "SUCCESS", "currency": "USD"},
    {"transaction_id": "TX_102", "amount": 2500.00, "status": "SUCCESS", "currency": "USD"},
    {"transaction_id": "TX_103", "amount": 45.10,   "status": "SUCCESS", "currency": "EUR"},
    {"transaction_id": "TX_104", "amount": 89.90,   "status": "SUCCESS", "currency": "USD"},
    {"transaction_id": "TX_105", "amount": 310.00,  "status": "SUCCESS", "currency": "GBP"},
    {"transaction_id": "TX_106", "amount": 75.50,   "status": "SUCCESS", "currency": "USD"},
    {"transaction_id": "TX_107", "amount": 1200.00, "status": "SUCCESS", "currency": "EUR"},
]

DEFAULT_LOGS = [
    {"event": "payment_processed",        "tx_id": "TX_101", "confirmed": True,  "latency_ms": 120},
    {"event": "payment_gateway_timeout",  "tx_id": "TX_102", "confirmed": False, "latency_ms": 9999},
    {"event": "payment_processed",        "tx_id": "TX_103", "confirmed": True,  "latency_ms": 88},
    {"event": "payment_processed",        "tx_id": "TX_104", "confirmed": True,  "latency_ms": 145},
    {"event": "payment_dropped_silently", "tx_id": "TX_105", "confirmed": False, "latency_ms": 0},
    {"event": "payment_processed",        "tx_id": "TX_106", "confirmed": True,  "latency_ms": 95},
    {"event": "payment_processed",        "tx_id": "TX_107", "confirmed": True,  "latency_ms": 200},
]

GROUND_TRUTH = {"TX_102", "TX_105"}


def compute_metrics(ground_truth: set, flagged: set) -> dict:
    tp = len(ground_truth & flagged)
    fp = len(flagged - ground_truth)
    fn = len(ground_truth - flagged)
    tn_proxy = len(DEFAULT_LEDGER) - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn_proxy) / len(DEFAULT_LEDGER) if len(DEFAULT_LEDGER) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn_proxy,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
    }


def render():
    st.title("🔄 Multi-Source Asynchronous Reconciliation Engine")
    st.markdown(
        "Cross-references a **structured CSV ledger** against **async JSON server logs** "
        "to surface silent transaction failures. Scores detection via Precision / Recall / F1."
    )
    st.divider()

    # ── Data Display ──────────────────────────────────────────────────────────
    col_csv, col_json = st.columns(2)

    with col_csv:
        st.markdown("#### 📄 Source A — Structured Ledger (CSV)")
        ledger_df = pd.DataFrame(DEFAULT_LEDGER)
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)
        st.caption(f"{len(ledger_df)} transactions — all marked SUCCESS in internal ledger")

    with col_json:
        st.markdown("#### 🌐 Source B — Async Server Logs (JSON)")
        logs_df = pd.DataFrame(DEFAULT_LOGS).rename(columns={"tx_id": "transaction_id"})
        # Color-code confirmed column
        def highlight_failed(row):
            if not row["confirmed"]:
                return ["background-color: #3a1a1a; color: #f85149"] * len(row)
            return [""] * len(row)
        st.dataframe(logs_df.style.apply(highlight_failed, axis=1), use_container_width=True, hide_index=True)
        st.caption("Red rows = unconfirmed events — potential silent failures")

    # ── Agent Simulation Controls ─────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🤖 Agent Anomaly Detection Configuration")

    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        detection_mode = st.selectbox(
            "Agent Detection Strategy",
            [
                "Confirmed=False flag only",
                "Latency threshold (>= 5000ms)",
                "Event keyword match (timeout/dropped)",
                "Combined: flag OR latency",
            ]
        )

    with col_cfg2:
        latency_threshold = st.slider("Latency Anomaly Threshold (ms)", 500, 30000, 5000, step=500)
        show_confusion = st.checkbox("Show Confusion Matrix", value=True)

    # ── Run Detection ─────────────────────────────────────────────────────────
    logs = DEFAULT_LOGS

    if detection_mode == "Confirmed=False flag only":
        flagged = {log["tx_id"] for log in logs if not log["confirmed"]}
    elif detection_mode == "Latency threshold (>= 5000ms)":
        flagged = {log["tx_id"] for log in logs if log["latency_ms"] >= latency_threshold}
    elif detection_mode == "Event keyword match (timeout/dropped)":
        flagged = {log["tx_id"] for log in logs if any(k in log["event"] for k in ["timeout", "dropped"])}
    else:  # Combined
        flagged = {
            log["tx_id"] for log in logs
            if not log["confirmed"] or log["latency_ms"] >= latency_threshold
        }

    metrics = compute_metrics(GROUND_TRUTH, flagged)

    # ── Results ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 Evaluation Results")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Precision", f"{metrics['precision']*100:.1f}%", help="Of flagged anomalies, how many were real?")
    col_m2.metric("Recall (Sensitivity)", f"{metrics['recall']*100:.1f}%", help="Of real anomalies, how many were caught?")
    col_m3.metric("F1-Score", f"{metrics['f1']:.4f}", help="Harmonic mean of Precision and Recall")
    col_m4.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")

    if metrics["f1"] == 1.0:
        st.success("🏆 PERFECT BENCHMARK — 100% reconciliation integrity. All anomalies correctly isolated, no false positives.")
    elif metrics["f1"] >= 0.75:
        st.warning(f"⚠️ PARTIAL PASS — F1={metrics['f1']:.4f}. Agent missed {metrics['fn']} anomaly(s) or raised {metrics['fp']} false alarm(s).")
    else:
        st.error(f"❌ BENCHMARK FAILED — F1={metrics['f1']:.4f}. Detection quality insufficient for production use.")

    # ── Flagged Transactions ───────────────────────────────────────────────────
    col_flag, col_conf_mat = st.columns(2)

    with col_flag:
        st.markdown("**Agent-Flagged Transactions**")
        flag_rows = []
        for log in logs:
            tx = log["tx_id"]
            is_flagged = tx in flagged
            is_anomaly = tx in GROUND_TRUTH
            if is_flagged:
                label = "TP ✅" if is_anomaly else "FP ⚠️"
                flag_rows.append({"TX ID": tx, "Event": log["event"], "Verdict": label})
        if flag_rows:
            st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Agent flagged no transactions under this strategy.")

        missed = GROUND_TRUTH - flagged
        if missed:
            st.error(f"**False Negatives (missed):** {', '.join(missed)}")

    with col_conf_mat:
        if show_confusion:
            st.markdown("**Confusion Matrix**")
            tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
            conf_df = pd.DataFrame(
                [[tp, fn], [fp, tn]],
                index=["Predicted: Anomaly", "Predicted: Normal"],
                columns=["Actual: Anomaly", "Actual: Normal"],
            )
            st.dataframe(conf_df, use_container_width=True)
            st.caption("TP=True Positive, FP=False Positive, FN=False Negative, TN=True Negative")

    # ── Payload Inspector ─────────────────────────────────────────────────────
    with st.expander("🔍 Raw JSON Log Inspector"):
        st.json(DEFAULT_LOGS)

    # ── Log to Session ────────────────────────────────────────────────────────
    st.session_state.scores["module3_f1"] = metrics["f1"]
    st.session_state.session_log.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] Module 3 ({detection_mode}) → F1={metrics['f1']:.4f}, "
        f"P={metrics['precision']:.2f}, R={metrics['recall']:.2f}"
    )
