# 🛡️ SwarmBench — AI Agent Evaluation Framework

> Enterprise-grade harness for benchmarking AI agents on statistical recovery, code safety, and multi-source data reconciliation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

SwarmBench is a modular evaluation framework for testing the analytical, safety, and coordination capabilities of AI data agents. Each module presents a realistic challenge with deterministic, verifiable scoring — the same principles used in production benchmarks like SWE-bench and Terminal-Bench.

---

## 🚀 Live Demo

**👉 [Launch SwarmBench on Streamlit](https://swarm-bench-cnxb5uqh4yfukisc6tsbc8.streamlit.app/)**

No setup required — runs entirely in the browser.

---

## Benchmark Modules

### 🧬 Module 1 — Distribution Mutation & Recovery
Tests whether an agent can restore a statistically corrupted dataset to its original distribution.

- Injects configurable **variance anomalies** and **null values** into a baseline distribution
- Supports Lognormal, Normal, Gamma, and Pareto distributions
- Scores recovery using **Kolmogorov-Smirnov Test**, **Wasserstein Distance**, and drift metrics
- Compares three recovery strategies: IQR clipping, Z-score clipping, Winsorization

**Pass Condition:** KS two-sample p-value > 0.05 (null hypothesis retained)

---

### 🛡️ Module 2 — Secure AST & Code Safety Evaluator
Verifies AI-generated Python code is safe before sandboxed execution.

- Static **Abstract Syntax Tree** scan across 16 banned identifiers (`os`, `sys`, `subprocess`, `eval`, etc.)
- Compiles verified code in a restricted `__builtins__`-free namespace
- Runs **5 parametric unit tests** including edge cases (all-negative weights, zero weights, float precision)
- Mirrors security constraints used in production AI code evaluation platforms

**Pass Condition:** Zero AST violations + all unit tests passed

---

### 🔄 Module 3 — Multi-Source Log Reconciliation
Evaluates an agent's ability to detect silent transaction failures across data sources.

- Cross-references a **structured CSV ledger** against **asynchronous JSON server logs**
- Supports 4 configurable detection strategies (confirmed flag, latency threshold, event keywords, combined)
- Scores using **Precision / Recall / F1-Score** with a full confusion matrix
- Surfaces true positives, false positives, and missed anomalies

**Pass Condition:** F1-Score = 1.0 (perfect isolation, no false alarms)

---

## Architecture

```
swarm_bench/
├── app.py                          # Main entry point & dashboard
├── pages/
│   ├── module1_distribution.py    # Distribution recovery benchmark
│   ├── module2_ast.py             # AST safety + code execution
│   └── module3_reconciliation.py  # Log reconciliation engine
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### Local (Python)
```bash
git clone https://github.com/ShubhSeth007/swarm-bench.git
cd swarm-bench
pip install -r requirements.txt
streamlit run app.py
```

### Docker (Recommended)
```bash
docker compose up --build
```
Then open [http://localhost:8501](http://localhost:8501).

---

## Design Principles

| Principle | Implementation |
|---|---|
| **Verifiable outcomes** | Every module has a deterministic pass/fail condition |
| **Configurable difficulty** | Sliders for corruption rate, sample size, thresholds |
| **Realistic data** | Messy CSVs, async JSON logs, real statistical distributions |
| **Safety-first execution** | AST scan before any `exec()` call |
| **Auditability** | Session log + JSON export for every run |

---

## Session Reporting

All benchmark runs are logged in the sidebar and exportable as a JSON report:

```json
{
  "generated_at": "2025-07-01T14:32:00Z",
  "scores": {
    "module1_ks": 0.0312,
    "module1_pval": 0.4218,
    "module2_passed": 5,
    "module2_total": 5,
    "module3_f1": 1.0
  },
  "log": [...]
}
```

---

## Tech Stack

- **Streamlit** — UI and interactive controls  
- **pandas / NumPy** — Data manipulation and generation  
- **SciPy** — KS-Test, Wasserstein Distance  
- **Python `ast` module** — Static code analysis  
- **Matplotlib** — Distribution visualization  
- **Docker** — Containerized deployment  

---

## License

MIT License — see [LICENSE](LICENSE) for details.
