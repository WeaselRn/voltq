"""Multi-seed recovery benchmark: same scenarios, same inputs, every strategy.

One seed shows an anecdote; this shows the distribution. All strategies see the identical post-event
state per (scenario, seed). Used by the dashboard and `run_recovery_benchmark.py`.
"""
from __future__ import annotations

import pandas as pd

from ..qaoa import QAOAConfig
from ..recovery.recovery_optimizer import STRATEGIES
from .network_simulator import NetworkSimulator
from .scenarios import SCENARIOS, build_scenario

COLS = ["recovery_runtime_s", "total_cost_inr", "avg_wait_min", "reassigned_reservations",
        "commitment_violations", "sla_success_pct", "critical_sla_pct", "unassigned_evs",
        "recovery_objective", "grid_power_violations"]


def run_benchmark(scenario_keys: list[str] | None = None, seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
                  n_evs: int = 12, strategies: tuple[str, ...] = STRATEGIES,
                  qcfg: QAOAConfig | None = None) -> pd.DataFrame:
    """Long table: one row per (scenario, seed, strategy)."""
    rows = []
    for key in scenario_keys or list(SCENARIOS):
        for seed in seeds:
            try:
                state, events = build_scenario(key, seed, n_evs=n_evs)
                sim = NetworkSimulator(state)
                run = sim.recover(sim.trigger(events), strategies, qcfg=qcfg or QAOAConfig(maxiter=30, shots=256))
            except (ValueError, KeyError, StopIteration):
                continue                                   # scenario not applicable for this seed
            for name, m in run.metrics.items():
                rows.append({"scenario": key, "seed": seed, "strategy": name, "valid": m["valid"],
                             "affected": len(run.report.affected), **{c: m[c] for c in COLS}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("strategy")[COLS + ["valid"]].mean(numeric_only=True)
    return g.reindex([s for s in STRATEGIES if s in g.index])
