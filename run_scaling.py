"""Scaling study: how do Greedy / CP-SAT / Hybrid(QAOA batches) change as the dataset grows?

  python run_scaling.py                 # default sizes
  python run_scaling.py 10 50 100 300   # custom EV counts
Writes outputs/scaling.csv and outputs/figures/scaling.png (real measured values only).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from voltq.classical import cpsat_schedule, greedy_schedule
from voltq.data import generate_scenario
from voltq.evaluate import evaluate_sessions
from voltq.hybrid import HybridConfig, hybrid_schedule
from voltq.models import ScenarioConfig
from voltq.preprocess import build_options
from voltq.qaoa import QAOAConfig

ROOT = Path(__file__).parent


def scenario_for(n_evs: int, seed: int = 42) -> ScenarioConfig:
    """Grow stations with fleet size (about 1 station per 4 EVs); ports 1-3; 24 slots."""
    return ScenarioConfig(n_evs=n_evs, n_stations=max(4, n_evs // 4), ports_min=1, ports_max=3,
                          n_slots=24, seed=seed)


def main() -> None:
    sizes = [int(x) for x in sys.argv[1:]] or [10, 25, 50, 100, 200]
    rows = []
    for n in sizes:
        scn = generate_scenario(scenario_for(n))
        pre = build_options(scn)
        t0 = time.perf_counter()
        runs = {"Greedy": greedy_schedule(scn, pre), "OR-Tools": cpsat_schedule(scn, pre, 30.0)}
        raw, final, logs = hybrid_schedule(scn, pre, QAOAConfig(maxiter=30, shots=256), HybridConfig())
        runs["Hybrid (QAOA batches)"], runs["Hybrid (+refinement)"] = raw, final
        for name, sch in runs.items():
            m = evaluate_sessions(scn, sch.sessions(scn.stations), pre.refs, sch.runtime_s)
            rows.append({
                "evs": n, "stations": len(scn.stations), "naive_vars": pre.full_size,
                "options": pre.reduced_size, "method": name, "valid": m["valid_partial"],
                "unassigned": m["unassigned_count"], "objective": round(m["objective"], 3),
                "cost_inr": round(m["total_cost_inr"]), "avg_wait_min": round(m["avg_wait_h"] * 60, 1),
                "runtime_s": round(sch.runtime_s, 2),
                "qaoa_calls": len(logs) if name.startswith("Hybrid") else 0,
                "cpsat_status": sch.meta.get("status", ""),
            })
        print(f"n={n} done in {time.perf_counter() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "outputs" / "scaling.csv", index=False)

    df = pd.DataFrame(rows)
    print(df.drop(columns=["naive_vars"]).to_string(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, g in df.groupby("method"):
        axes[0].plot(g["evs"], g["objective"], marker="o", label=name)
        axes[1].plot(g["evs"], g["runtime_s"], marker="o", label=name)
        axes[2].plot(g["evs"], g["unassigned"], marker="o", label=name)
    for ax, t in zip(axes, ["Objective (lower = better)", "Runtime (s)", "EVs left uncharged"]):
        ax.set_title(t)
        ax.set_xlabel("number of EVs")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "figures" / "scaling.png", dpi=130)


if __name__ == "__main__":
    main()
