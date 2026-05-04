# QPO — Quantum Prompt Optimisation

Research project exploring QUBO/QAOA formulations for prompt feature selection. The core question: can a quantum algorithm find better prompt feature combinations than classical greedy search at equivalent evaluation cost?

**Results writeup:** [waratahlabs.github.io/2026/05/03/qpo-preliminary-findings/](https://waratahlabs.github.io/2026/05/03/qpo-preliminary-findings/)

Phase 1 runs entirely on a CUDA-accelerated simulator (PennyLane `lightning.gpu`). Next phases: larger circuits approaching classical simulation limits (~50 qubits), then physical quantum hardware.

---

## How it works

Prompt quality is a combinatorial problem — feature interactions determine output quality in ways a linear scorer can't capture. QPO formulates feature selection as a QUBO (Quadratic Unconstrained Binary Optimisation) problem and solves it with QAOA, running the combinatorial search over the feature space rather than greedy approximation.

The pipeline has five stages:

```
Intent (plain English goal)
  ↓
1. Decomposer       — identifies 6–14 feature axes from the goal
2. CandidateGenerator — expands axes into prompt variants (capped at max_candidates)
3. PreScorer        — fast model scores all candidates (granite4.1:3b)
4. QuantumOptimizer — QUBO formulation + QAOA shortlists top-K combination
5. DeepEvaluator    — scores QAOA shortlist and classical baseline (granite4.1:8b)
  ↓
PipelineRun (winning variant, score, classical comparison, latencies)
```

The classical baseline (top-K by pre-score rank) is evaluated alongside every QAOA run in a single deduped inference pass. Results include QAOA score, classical score, delta, and candidate overlap.

---

## Phase 1 results (n=50)

50 runs across 5 goals (10 runs each). `circuit_depth=3`, `num_iterations=50`, 20 qubits.

| Goal | QAOA wins | Classical wins | Ties | Mean Δ |
|------|-----------|----------------|------|--------|
| JSON extraction | 0 | 1 | 9 | −0.005 |
| Ticket classification | 1 | 0 | 9 | +0.005 |
| **Git commit message** | **5** | **1** | **4** | **+0.030** |
| CVE risk assessment | 1 | 2 | 7 | −0.005 |
| Legal clause rewrite | 1 | 1 | 8 | 0.000 |
| **Total** | **8** | **5** | **37** | **+0.005** |

QAOA advantage concentrates in tasks with open-ended answer spaces (git commit: 5 wins, +0.030 mean delta). Constrained-output tasks converge to near-identical shortlists. Full interpretation in the [writeup](https://waratahlabs.github.io/2026/05/03/qpo-preliminary-findings/).

---

## Hardware

- **Quantum simulation:** Predator (Intel U9-275HX, RTX 5070Ti Mobile 12GB VRAM) — PennyLane `lightning.gpu` via WSL2
- **Pre-scorer:** Mac (M1 Pro) — `granite4.1:3b` via Ollama
- **Deep evaluator:** Mac (M1 Pro) — `granite4.1:8b` via Ollama

---

## Getting started

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Single run
python -m qpo --goal "Write a conventional commit message for a given diff"

# Web UI (with SSE streaming)
python -m qpo.server

# Run tests
pytest tests/ -v
```

### Configuration

All configuration via environment variables (see `.env.predator` for reference):

| Variable | Default | Description |
|----------|---------|-------------|
| `QPO_OLLAMA_ENDPOINT` | `http://localhost:11434` | Ollama endpoint |
| `QPO_MODEL_7B` | `granite4.1:3b` | Pre-scorer model |
| `QPO_MODEL_32B` | `granite4.1:8b` | Deep evaluator model |
| `QPO_QUANTUM_BACKEND` | `stub` | `stub`, `lightning`, or `lightning.gpu` |
| `QPO_CIRCUIT_DEPTH` | `3` | QAOA circuit depth |
| `QPO_QAOA_ITERATIONS` | `50` | QAOA optimisation iterations |
| `QPO_OLLAMA_TIMEOUT` | `180` | Per-call timeout (seconds) |

---

## Web UI

`python -m qpo.server` starts a Flask server at `http://localhost:5000` with:
- Single run mode with live SSE event streaming
- Batch run mode (multiple goals × N runs) with aggregate QAOA vs classical results
- CSV export
- Run history

---

## Project

[Waratah Labs](https://waratahlabs.github.io) · [writeup](https://waratahlabs.github.io/2026/05/03/qpo-preliminary-findings/)
