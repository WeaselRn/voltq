"""QAOA on the local Qiskit Aer simulator (shot-based sampling, no IBM hardware/cloud).

Workflow per objective evaluation:
  1. bind (gamma, beta) into the QAOA circuit built from the QUBO's Ising Hamiltonian,
  2. sample `shots` bitstrings on AerSimulator,
  3. decode bitstrings and compute their QUBO energy,
  4. return the sample-mean (or CVaR) energy to a classical optimizer (scipy).
Every evaluation is logged, so the progress chart is real optimization history.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .qubo import QUBO, to_ising

OPTIMIZERS = ("COBYLA", "Nelder-Mead", "Powell")


@dataclass
class QAOAConfig:
    depth: int = 1
    shots: int = 512
    optimizer: str = "COBYLA"
    maxiter: int = 60
    seed: int = 7
    cvar_alpha: float = 1.0       # 1.0 = plain expectation; <1 = CVaR of the best alpha-fraction
    restarts: int = 1
    final_shots_mult: int = 4     # extra shots for the final distribution


@dataclass
class QAOAResult:
    config: QAOAConfig
    n_qubits: int
    circuit_depth: int
    gate_counts: dict
    n_evaluations: int
    sim_time_s: float
    best_energy: float                      # best energy of any measured bitstring
    best_bits: np.ndarray | None
    best_feasible_cost: float | None        # decision-variable objective of best feasible sample
    best_feasible_bits: np.ndarray | None
    history: list[dict] = field(default_factory=list)
    final_counts: dict = field(default_factory=dict)       # bitstring (decision vars) -> shots
    final_expected_energy: float = float("nan")
    params: np.ndarray | None = None
    prob_feasible: float = 0.0
    prob_best_feasible: float = 0.0


def build_circuit(q: QUBO, p: int) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    h, J, _ = to_ising(q)
    scale = max([abs(x) for x in h] + [abs(v) for v in J.values()] + [1e-12])
    gam = [Parameter(f"g{k}") for k in range(p)]
    bet = [Parameter(f"b{k}") for k in range(p)]
    qc = QuantumCircuit(q.n)
    qc.h(range(q.n))
    for k in range(p):
        for i in range(q.n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * gam[k] * (h[i] / scale), i)
        for (i, j), v in J.items():
            if abs(v) > 1e-12:
                qc.rzz(2 * gam[k] * (v / scale), i, j)
        for i in range(q.n):
            qc.rx(2 * bet[k], i)
    qc.measure_all()
    return qc, gam, bet


def _bits(key: str, n: int) -> np.ndarray:
    key = key.replace(" ", "")
    return np.array([int(key[n - 1 - i]) for i in range(n)], dtype=int)   # little-endian


def _init_params(p: int, rng: np.random.Generator, jitter: float) -> np.ndarray:
    """Linear-ramp (annealing-inspired) start plus seeded jitter."""
    frac = (np.arange(p) + 0.5) / p
    g = 0.8 * frac
    b = 0.8 * (1 - frac)
    x = np.concatenate([g, b])
    return x + rng.normal(0, jitter, size=2 * p)


def run_qaoa(q: QUBO, cfg: QAOAConfig) -> QAOAResult:
    if q.n == 0:
        raise ValueError("empty QUBO")
    t0 = time.perf_counter()
    qc, gam, bet = build_circuit(q, cfg.depth)
    backend = AerSimulator()
    tqc = transpile(qc, backend, optimization_level=1, seed_transpiler=cfg.seed)
    params = list(tqc.parameters)
    order = {p.name: p for p in params}
    gb = [order[g.name] for g in gam] + [order[b.name] for b in bet]

    energy_cache: dict[str, float] = {}
    feas_cache: dict[str, bool] = {}
    state = {"evals": 0, "best_e": np.inf, "best_b": None, "bf_e": np.inf, "bf_b": None}
    history: list[dict] = []

    def sample(x: np.ndarray, shots: int) -> dict[str, int]:
        bound = tqc.assign_parameters(dict(zip(gb, x)))
        res = backend.run(bound, shots=shots, seed_simulator=cfg.seed).result()
        return res.get_counts()

    def info(key: str) -> tuple[float, bool]:
        if key not in energy_cache:
            b = _bits(key, q.n)
            energy_cache[key] = q.energy(b)
            feas_cache[key] = q.is_feasible(b)
        return energy_cache[key], feas_cache[key]

    def track(counts: dict[str, int]) -> None:
        for key in counts:
            e, f = info(key)
            if e < state["best_e"]:
                state["best_e"], state["best_b"] = e, _bits(key, q.n)
            if f:
                b = _bits(key, q.n)
                c = q.decision_cost(b)      # true objective of the decoded schedule
                if c < state["bf_e"]:
                    state["bf_e"], state["bf_b"] = c, b

    def objective(x: np.ndarray, restart: int = 0) -> float:
        counts = sample(x, cfg.shots)
        track(counts)
        es = np.array([info(k)[0] for k in counts])
        w = np.array([counts[k] for k in counts], dtype=float)
        order_ = np.argsort(es)
        es, w = es[order_], w[order_]
        if cfg.cvar_alpha < 1.0:
            cutoff = cfg.cvar_alpha * w.sum()
            cum = np.cumsum(w)
            take = np.clip(w - np.maximum(cum - cutoff, 0), 0, None)
            val = float((es * take).sum() / take.sum())
        else:
            val = float((es * w).sum() / w.sum())
        state["evals"] += 1
        history.append({"iter": state["evals"], "restart": restart, "expected_energy": val,
                        "best_energy": state["best_e"],
                        "best_feasible_cost": None if state["bf_b"] is None else state["bf_e"]})
        return val

    rng = np.random.default_rng(cfg.seed)
    best_x, best_val = None, np.inf
    for r in range(max(1, cfg.restarts)):
        x0 = _init_params(cfg.depth, rng, 0.0 if r == 0 else 0.3)
        budget = max(5, cfg.maxiter // max(1, cfg.restarts))
        opts = {"maxiter": budget}
        if cfg.optimizer == "COBYLA":
            opts["rhobeg"] = 0.4
        res = minimize(lambda x, r=r: objective(x, r), x0, method=cfg.optimizer, options=opts)
        if res.fun < best_val:
            best_val, best_x = float(res.fun), res.x

    final = sample(best_x, cfg.shots * cfg.final_shots_mult)
    track(final)
    tot = sum(final.values())
    exp_e = sum(info(k)[0] * c for k, c in final.items()) / tot
    p_feas = sum(c for k, c in final.items() if info(k)[1]) / tot
    p_bf = 0.0
    if state["bf_b"] is not None:     # slack bits may differ, so match on decision bits only
        nd = q.n_decision
        p_bf = sum(c for k, c in final.items()
                   if np.array_equal(_bits(k, q.n)[:nd], state["bf_b"][:nd])) / tot
    ops = dict(tqc.count_ops())
    ops.pop("measure", None)
    ops.pop("barrier", None)
    return QAOAResult(
        config=cfg, n_qubits=q.n, circuit_depth=tqc.depth(), gate_counts=ops,
        n_evaluations=state["evals"], sim_time_s=time.perf_counter() - t0,
        best_energy=state["best_e"], best_bits=state["best_b"],
        best_feasible_cost=None if state["bf_b"] is None else state["bf_e"],
        best_feasible_bits=state["bf_b"], history=history,
        final_counts=dict(sorted(final.items(), key=lambda kv: -kv[1])),
        final_expected_energy=exp_e, params=best_x, prob_feasible=p_feas, prob_best_feasible=p_bf)

