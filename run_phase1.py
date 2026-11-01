"""Phase 1 demo: data -> preprocessing -> baselines -> QAOA (Aer) on reduced instance -> outputs.

Run:  python run_phase1.py [--seed 42] [--depth 1] [--shots 512] [--maxiter 60]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from voltq import viz
from voltq.data import generate_scenario, save_scenario
from voltq.models import ScenarioConfig
from voltq.pipeline import (evaluate_all, jsonable, qaoa_summary, run_baselines,
                            run_reduced_instance, schedule_table)
from voltq.preprocess import build_options, distance_matrix
from voltq.qaoa import QAOAConfig

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evs", type=int, default=10)
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    a = ap.parse_args()

    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    scn = generate_scenario(ScenarioConfig(n_evs=a.evs, n_stations=a.stations, seed=a.seed))
    pre = build_options(scn)
    save_scenario(scn, ROOT / "data", distance_matrix(scn))
    print(f"[data] {len(scn.evs)} EVs, {len(scn.stations)} stations, {scn.config.n_slots} slots -> data/")
    print(f"[preprocess] naive variables {pre.full_size} -> feasible options {pre.reduced_size}; "
          f"infeasible EVs: {pre.infeasible_evs or 'none'}")

    schedules = run_baselines(scn, pre)
    evals = evaluate_all(scn, pre, schedules)
    for name, m in evals.items():
        print(f"[{name}] feasible={m['feasible']} cost=EUR {m['total_cost_eur']:.2f} "
              f"avg_wait={m['avg_wait_h']:.2f}h unassigned={m['unassigned_count']} "
              f"runtime={m['runtime_s']:.3f}s")

    qcfg = QAOAConfig(depth=a.depth, shots=a.shots, maxiter=a.maxiter)
    red = run_reduced_instance(scn, pre, qcfg)
    qr = red["qaoa"]
    print(f"[QAOA/Aer] reduced instance: EVs {red['ev_ids']} -> {red['n_variables']} qubits "
          f"({red['n_decision']} decision + {red['n_slack']} slack), p={qcfg.depth}, "
          f"{qr.n_evaluations} circuit evaluations, circuit depth {qr.circuit_depth}, {qr.sim_time_s:.2f}s")
    print(f"           objective by method: {red['objective_by_method']}")

    for name, s in schedules.items():
        schedule_table(scn, s).to_csv(out / f"schedule_{name.lower().replace('-', '').replace(' ', '_')}.csv", index=False)
    best = min((n for n in evals if evals[n]["feasible"]), key=lambda n: evals[n]["objective"])
    schedule_table(scn, schedules[best]).to_csv(out / "optimized_schedule.csv", index=False)

    summary = {
        "scenario": {"n_evs": len(scn.evs), "n_stations": len(scn.stations), "seed": scn.config.seed},
        "preprocessing": {"naive_variables": pre.full_size, "feasible_options": pre.reduced_size},
        "full_scenario_methods": {n: {k: v for k, v in m.items() if k != "occupancy"} for n, m in evals.items()},
        "reduced_instance": {
            "ev_ids": red["ev_ids"], "n_variables": red["n_variables"],
            "n_decision": red["n_decision"], "n_slack": red["n_slack"],
            "objective_by_method": red["objective_by_method"],
            "qaoa_matches_exact": red["qaoa_matches_exact"],
            "qaoa": qaoa_summary(qr),
            "qaoa_history": qr.history,
        },
        "note": "QAOA ran only on the reduced instance; the 10-EV schedules come from classical solvers.",
    }
    (out / "comparison.json").write_text(json.dumps(jsonable(summary), indent=2))

    figs = out / "figures"
    viz.fig_tariff(scn).savefig(figs / "tariff.png", dpi=130)
    viz.fig_schedule(scn, schedules[best].sessions(scn.stations), f"Charging schedule ({best})").savefig(figs / "assignment.png", dpi=130)
    viz.fig_utilization(scn, evals[best]).savefig(figs / "utilization.png", dpi=130)
    ex = red["exact"]["min_energy"] if red["exact"] else None
    viz.fig_qaoa_progress(qr.history, ex).savefig(figs / "qaoa_progress.png", dpi=130)
    viz.fig_comparison(evals, "Greedy vs OR-Tools (full 10-EV scenario)").savefig(figs / "comparison.png", dpi=130)
    viz.fig_reduced_comparison(red["objective_by_method"]).savefig(figs / "quantum_vs_classical.png", dpi=130)
    print(f"[output] CSV/JSON in {out}, figures in {figs}")


if __name__ == "__main__":
    main()
