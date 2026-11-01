"""End-to-end orchestration used by the CLI and the dashboard."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .classical import (committed_residual, cpsat_schedule, fits, full_residual, greedy_schedule,
                        solve_cpsat)
from .data import generate_scenario, save_scenario
from .evaluate import evaluate_sessions
from .models import Option, Scenario, ScenarioConfig, Schedule
from .preprocess import Preprocessed, build_options, distance_matrix
from .qaoa import QAOAConfig, QAOAResult, run_qaoa
from .qubo import QUBO, build_qubo, exact_solve


def reduce_options(opts: list[Option], k: int) -> list[Option]:
    """Keep the k cheapest options of one EV (classical search-space reduction)."""
    return sorted(opts, key=lambda o: (o.cost, o.start))[:k]


def pick_demonstrator(scn: Scenario, pre: Preprocessed, n_ev: int = 3, k: int = 3,
                      max_qubits: int = 16) -> dict[int, list[Option]]:
    """Reduced instance for QAOA: the n_ev EVs most in contention for one station,
    each limited to its k cheapest options. Shrinks k until the QUBO has <= max_qubits."""
    best_station: dict[int, int] = {}
    for ev_id, opts in pre.options.items():
        if opts:
            best_station[ev_id] = min(opts, key=lambda o: o.cost).station_id
    by_station: dict[int, list[int]] = {}
    for ev_id, s in best_station.items():
        by_station.setdefault(s, []).append(ev_id)
    # EVs sharing a favourite station, then fill with the next most similar
    group = max(by_station.values(), key=len) if by_station else []
    chosen = sorted(group, key=lambda i: scn.evs[i].arrival_slot)[:n_ev]
    for ev_id in sorted(best_station, key=lambda i: scn.evs[i].arrival_slot):
        if len(chosen) >= n_ev:
            break
        if ev_id not in chosen:
            chosen.append(ev_id)
    for kk in range(k, 0, -1):
        sub = {i: reduce_options(pre.options[i], kk) for i in chosen}
        if build_qubo(sub, full_residual(scn)).n <= max_qubits:
            return sub
    return {i: reduce_options(pre.options[i], 1) for i in chosen}


def _greedy_sub(scn: Scenario, sub: dict[int, list[Option]]) -> dict[int, Option]:
    got: dict[int, Option] = {}
    for ev_id in sorted(sub, key=lambda i: scn.evs[i].departure_slot - scn.evs[i].arrival_slot):
        res = committed_residual(scn, got)
        cand = [o for o in sub[ev_id] if fits(o, res)]
        if cand:
            got[ev_id] = min(cand, key=lambda o: o.cost)
    return got


def run_reduced_instance(scn: Scenario, pre: Preprocessed, qcfg: QAOAConfig,
                         sub: dict[int, list[Option]] | None = None) -> dict:
    """QAOA on the reduced demonstrator, compared with exact/CP-SAT/greedy on the SAME options."""
    sub = sub or pick_demonstrator(scn, pre)
    q = build_qubo(sub, full_residual(scn))
    pen = scn.config.unassigned_penalty
    res = run_qaoa(q, qcfg)
    exact = exact_solve(q) if q.n <= 22 else None

    def total(assign: dict[int, Option]) -> float:
        return sum(o.cost for o in assign.values()) + pen * (len(sub) - len(assign))

    qaoa_assign = ({o.ev_id: o for o in q.decode(res.best_feasible_bits)}
                   if res.best_feasible_bits is not None else {})
    cp, status, _ = solve_cpsat(sub, full_residual(scn), pen)
    gr = _greedy_sub(scn, sub)
    exact_assign = ({o.ev_id: o for o in q.decode(exact["min_feasible_bits"])}
                    if exact and exact["min_feasible_bits"] is not None else None)
    table = {
        "QAOA (Aer, best feasible sample)": total(qaoa_assign) if qaoa_assign else None,
        "Exact QUBO optimum (brute force)": total(exact_assign) if exact_assign else None,
        "OR-Tools CP-SAT": total(cp),
        "Greedy": total(gr),
    }
    energies = {k: v for k, v in ((k, None) for k in res.final_counts)}
    for k in res.final_counts:
        b = np.array([int(k[q.n - 1 - i]) for i in range(q.n)])
        energies[k] = (q.energy(b), q.is_feasible(b))
    return {
        "ev_ids": list(sub), "n_variables": q.n, "n_decision": q.n_decision, "n_slack": q.n_slack,
        "n_capacity_constraints": len(q.capacity), "penalties": q.penalties,
        "options_per_ev": {i: len(v) for i, v in sub.items()},
        "qubo": q, "qaoa": res, "exact": exact, "objective_by_method": table,
        "cpsat_status": status, "bit_info": energies,
        "qaoa_assignment": qaoa_assign,
        "qaoa_matches_exact": (exact_assign is not None and bool(qaoa_assign)
                               and abs(total(qaoa_assign) - total(exact_assign)) < 1e-9),
    }


def run_baselines(scn: Scenario, pre: Preprocessed, cp_time: float = 20.0) -> dict[str, Schedule]:
    return {"Greedy": greedy_schedule(scn, pre), "OR-Tools": cpsat_schedule(scn, pre, cp_time)}


def evaluate_all(scn: Scenario, pre: Preprocessed, schedules: dict[str, Schedule]) -> dict[str, dict]:
    return {name: evaluate_sessions(scn, s.sessions(scn.stations), pre.refs, s.runtime_s)
            for name, s in schedules.items()}


def schedule_table(scn: Scenario, schedule: Schedule) -> pd.DataFrame:
    h = scn.config.slot_hours
    sessions = {s.ev_id: s for s in schedule.sessions(scn.stations)}
    rows = []
    for ev in scn.evs:
        s, o = sessions.get(ev.ev_id), schedule.assignments.get(ev.ev_id)
        rows.append({
            "ev": ev.name, "status": "assigned" if s else "UNASSIGNED",
            "station": scn.stations[s.station_id].name if s else "-",
            "port": s.port + 1 if s else None,
            "start_slot": s.start if s else None, "end_slot": s.start + s.duration if s else None,
            "start_h": round(scn.config.start_hour + s.start * h, 2) if s else None,
            "required_kwh": round(o.energy_kwh, 2) if o else None,
            "energy_cost_eur": round(o.energy_cost, 2) if o else None,
            "wait_h": o.wait_h if o else None,
            "distance_km": round(o.distance_km, 2) if o else None,
        })
    return pd.DataFrame(rows)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def qaoa_summary(res: QAOAResult) -> dict:
    return {
        "depth": res.config.depth, "shots": res.config.shots, "optimizer": res.config.optimizer,
        "maxiter": res.config.maxiter, "seed": res.config.seed, "cvar_alpha": res.config.cvar_alpha,
        "qubits": res.n_qubits, "circuit_depth": res.circuit_depth, "gate_counts": res.gate_counts,
        "gate_total": int(sum(res.gate_counts.values())),
        "evaluations": res.n_evaluations, "sim_time_s": round(res.sim_time_s, 3),
        "best_measured_energy": res.best_energy, "best_feasible_cost": res.best_feasible_cost,
        "best_candidate_feasible": res.best_feasible_bits is not None,
        "prob_feasible_final": res.prob_feasible, "prob_best_feasible_final": res.prob_best_feasible,
        "final_expected_energy": res.final_expected_energy,
    }
