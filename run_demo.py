"""Phase 2 demo: hybrid QAOA scheduling, depth sweep, all comparisons and outputs.

  python run_demo.py
  python run_demo.py --evs 15 --stations 5 --ports-min 1 --ports-max 3 --depth 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from voltq.hybrid import HybridConfig
from voltq.models import ScenarioConfig
from voltq.qaoa import OPTIMIZERS, QAOAConfig
from voltq.study import run_all, save_outputs

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evs", type=int, default=10)
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--ports-min", type=int, default=2)
    ap.add_argument("--ports-max", type=int, default=2)
    ap.add_argument("--slots", type=int, default=12)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--optimizer", default="COBYLA", choices=OPTIMIZERS)
    ap.add_argument("--cvar", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    a = ap.parse_args()

    cfg = ScenarioConfig(n_evs=a.evs, n_stations=a.stations, ports_min=a.ports_min,
                         ports_max=a.ports_max, n_slots=a.slots, seed=a.seed)
    qcfg = QAOAConfig(depth=a.depth, shots=a.shots, optimizer=a.optimizer, maxiter=a.maxiter,
                      seed=a.seed, cvar_alpha=a.cvar)
    res = run_all(cfg, qcfg, HybridConfig(batch_size=a.batch_size), do_sweep=not a.no_sweep)
    save_outputs(res, Path(a.out), ROOT / "data")

    pre = res["pre"]
    print(f"EVs={a.evs} stations={a.stations}: naive vars {pre.full_size} -> {pre.reduced_size} options")
    print(f"{'method':30s} {'feasible':8s} {'EUR':>8s} {'objective':>9s} {'avg wait h':>10s} {'unassigned':>10s} {'runtime s':>9s}")
    for n, m in res["evals"].items():
        print(f"{n:30s} {str(m['feasible']):8s} {m['total_cost_eur']:8.2f} {m['objective']:9.3f} "
              f"{m['avg_wait_h']:10.2f} {m['unassigned_count']:10d} {m['runtime_s']:9.2f}")
    print("hybrid batches:")
    for l in res["hybrid_logs"]:
        q = l.qaoa
        extra = "" if q is None else f"evals={q.n_evaluations} t={q.sim_time_s:.1f}s"
        print(f"  batch {l.batch}: EVs {l.ev_ids} qubits={l.n_qubits} fallback={l.used_fallback} {extra} {l.note}")
    red = res["reduced"]
    print("reduced instance:", red["objective_by_method"], "\n  random control:", red["random_control"])
    for r in res["depth_sweep"]:
        print("  depth sweep:", r)


if __name__ == "__main__":
    main()
