"""Full study: all schedulers + reduced-instance QAOA experiments + output writer."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import viz
from .data import generate_scenario, save_scenario
from .hybrid import HybridConfig, hybrid_schedule
from .models import Option, Scenario, ScenarioConfig
from .pipeline import (evaluate_all, jsonable, pick_demonstrator, qaoa_summary, run_baselines,
                       run_reduced_instance, schedule_table)
from .preprocess import Preprocessed, build_options, distance_matrix
from .qaoa import QAOAConfig
from .qubo import QUBO


def random_control(q: QUBO, n_samples: int, seed: int, best_cost: float | None = None) -> dict:
    """Uniform random bitstrings with the same total shot budget as QAOA.

    Honest control: if random sampling finds the same answer, QAOA shows no benefit here.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_samples, q.n))
    best, n_feas, n_best = None, 0, 0
    for row in X:
        if q.is_feasible(row):
            n_feas += 1
            c = q.decision_cost(row)
            if best is None or c < best:
                best = c
            if best_cost is not None and c <= best_cost + 1e-9:
                n_best += 1
    return {"samples": n_samples, "prob_feasible": n_feas / n_samples, "best_feasible_cost": best,
            "prob_best": n_best / n_samples}


def depth_sweep(scn: Scenario, pre: Preprocessed, base: QAOAConfig, depths=(1, 2, 3),
                sub: dict[int, list[Option]] | None = None) -> list[dict]:
    """Reduced instance at several QAOA depths; one summary row per depth."""
    sub = sub or pick_demonstrator(scn, pre)
    rows = []
    for p in depths:
        r = run_reduced_instance(scn, pre, replace(base, depth=p), sub)
        s = qaoa_summary(r["qaoa"])
        rows.append({
            "depth_p": p, "qubits": s["qubits"], "circuit_depth": s["circuit_depth"],
            "gates": s["gate_total"], "evaluations": s["evaluations"], "sim_time_s": s["sim_time_s"],
            "best_feasible_cost": s["best_feasible_cost"],
            "exact_cost": r["objective_by_method"]["Exact QUBO optimum (brute force)"],
            "prob_feasible": s["prob_feasible_final"], "prob_best": s["prob_best_feasible_final"],
            "found_optimum": r["qaoa_matches_exact"],
        })
    return rows


def run_all(cfg: ScenarioConfig, qcfg: QAOAConfig, hcfg: HybridConfig | None = None,
            depths=(1, 2, 3), cp_time: float = 20.0, do_sweep: bool = True) -> dict:
    scn = generate_scenario(cfg)
    pre = build_options(scn)
    schedules = run_baselines(scn, pre, cp_time)
    raw, final, logs = hybrid_schedule(scn, pre, qcfg, hcfg)
    schedules[raw.name] = raw
    schedules[final.name] = final
    evals = evaluate_all(scn, pre, schedules)
    red = run_reduced_instance(scn, pre, qcfg)
    red["random_control"] = random_control(
        red["qubo"], red["qaoa"].n_evaluations * qcfg.shots, qcfg.seed,
        red["objective_by_method"]["Exact QUBO optimum (brute force)"])
    sweep = depth_sweep(scn, pre, qcfg, depths) if do_sweep else []
    return {"scenario": scn, "pre": pre, "schedules": schedules, "evals": evals,
            "reduced": red, "hybrid_logs": logs, "depth_sweep": sweep}


def _slug(n: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in n).strip("_")


def best_method(evals: dict) -> str:
    ok = [n for n in evals if evals[n]["feasible"]]
    return min(ok or list(evals), key=lambda n: evals[n]["objective"])


def save_outputs(res: dict, out: Path, data_dir: Path) -> None:
    scn, pre, schedules, evals, red = (res[k] for k in ("scenario", "pre", "schedules", "evals", "reduced"))
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    save_scenario(scn, data_dir, distance_matrix(scn))
    for name, s in schedules.items():
        schedule_table(scn, s).to_csv(out / f"schedule_{_slug(name)}.csv", index=False)
    best = best_method(evals)
    schedule_table(scn, schedules[best]).to_csv(out / "optimized_schedule.csv", index=False)
    qr = red["qaoa"]
    summary = {
        "scenario": {"n_evs": len(scn.evs), "n_stations": len(scn.stations),
                     "n_slots": scn.config.n_slots, "seed": scn.config.seed},
        "preprocessing": {"naive_variables": pre.full_size, "feasible_options": pre.reduced_size,
                          "infeasible_evs": pre.infeasible_evs, "refs": pre.refs},
        "full_scenario_methods": {n: {k: v for k, v in m.items() if k != "occupancy"} for n, m in evals.items()},
        "hybrid_batches": [{"batch": l.batch, "ev_ids": l.ev_ids, "qubits": l.n_qubits,
                            "fallback": l.used_fallback, "committed": l.committed, "note": l.note,
                            "qaoa": qaoa_summary(l.qaoa) if l.qaoa else None} for l in res["hybrid_logs"]],
        "reduced_instance": {
            "ev_ids": red["ev_ids"], "n_variables": red["n_variables"],
            "objective_by_method": red["objective_by_method"],
            "qaoa_matches_exact": red["qaoa_matches_exact"],
            "random_control": red["random_control"],
            "qaoa": qaoa_summary(qr), "qaoa_history": qr.history},
        "depth_sweep": res["depth_sweep"],
        "best_feasible_method": best,
        "note": "QAOA runs only on reduced instances / batches on Aer; no quantum advantage is claimed.",
    }
    (out / "comparison.json").write_text(json.dumps(jsonable(summary), indent=2))
    viz.fig_tariff(scn).savefig(figs / "tariff.png", dpi=130)
    viz.fig_schedule(scn, schedules[best].sessions(scn.stations), f"Charging schedule ({best})").savefig(figs / "assignment.png", dpi=130)
    viz.fig_utilization(scn, evals[best], f"Station utilization ({best})").savefig(figs / "utilization.png", dpi=130)
    ex = red["exact"]["min_energy"] if red["exact"] else None
    viz.fig_qaoa_progress(qr.history, ex).savefig(figs / "qaoa_progress.png", dpi=130)
    viz.fig_reduced_comparison(red["objective_by_method"]).savefig(figs / "quantum_vs_classical.png", dpi=130)
    viz.fig_comparison(evals, "Full scenario: all schedulers").savefig(figs / "comparison.png", dpi=130)
