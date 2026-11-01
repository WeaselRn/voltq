"""Classical solvers: greedy heuristic and OR-Tools CP-SAT (same objective & constraints)."""
from __future__ import annotations

import time
from typing import Callable

from ortools.sat.python import cp_model

from .models import Option, Scenario, Schedule
from .preprocess import Preprocessed

Residual = Callable[[int, int], int]
SCALE = 10_000   # CP-SAT needs integer costs


def full_residual(scn: Scenario) -> Residual:
    ports = {s.station_id: s.ports for s in scn.stations}
    return lambda s, t: ports[s]


def committed_residual(scn: Scenario, committed: dict[int, Option]) -> Residual:
    ports = {s.station_id: s.ports for s in scn.stations}
    use: dict[tuple[int, int], int] = {}
    for o in committed.values():
        for t in o.slots:
            use[(o.station_id, t)] = use.get((o.station_id, t), 0) + 1
    return lambda s, t: ports[s] - use.get((s, t), 0)


def fits(o: Option, residual: Residual) -> bool:
    return all(residual(o.station_id, t) >= 1 for t in o.slots)


def greedy_schedule(scn: Scenario, pre: Preprocessed) -> Schedule:
    """Earliest-deadline-first; each EV takes its cheapest option that still fits."""
    t0 = time.perf_counter()
    evs = sorted(scn.evs, key=lambda e: (e.departure_slot - e.arrival_slot, e.ev_id))
    assigned: dict[int, Option] = {}
    for ev in evs:
        res = committed_residual(scn, assigned)
        cand = [o for o in pre.options[ev.ev_id] if fits(o, res)]
        if cand:
            assigned[ev.ev_id] = min(cand, key=lambda o: (o.cost, o.start))
    unassigned = [e.ev_id for e in scn.evs if e.ev_id not in assigned]
    return Schedule("Greedy", assigned, unassigned, time.perf_counter() - t0)


def solve_cpsat(options_by_ev: dict[int, list[Option]], residual: Residual,
                unassigned_penalty: float, time_limit: float = 20.0, seed: int = 1,
                ) -> tuple[dict[int, Option], str, float]:
    """Max-service min-cost assignment. Returns (assignments, status, objective)."""
    m = cp_model.CpModel()
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    opts: list[Option] = []
    cover: dict[tuple[int, int], list] = {}
    unassigned_vars = []
    obj = []
    for ev_id, lst in options_by_ev.items():
        ev_vars = []
        for o in lst:
            if not fits(o, residual):
                continue
            v = m.NewBoolVar(f"x_{len(opts)}")
            x[(len(opts), ev_id)] = v
            opts.append(o)
            ev_vars.append(v)
            obj.append(int(round(o.cost * SCALE)) * v)
            for t in o.slots:
                cover.setdefault((o.station_id, t), []).append(v)
        u = m.NewBoolVar(f"u_{ev_id}")
        m.Add(sum(ev_vars) + u == 1)
        unassigned_vars.append(u)
        obj.append(int(round(unassigned_penalty * SCALE)) * u)
    for (s, t), vs in cover.items():
        m.Add(sum(vs) <= residual(s, t))
    m.Minimize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = 4
    status = solver.Solve(m)
    name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {}, name, float("nan")
    chosen = {}
    for (k, ev_id), v in x.items():
        if solver.Value(v):
            chosen[ev_id] = opts[k]
    return chosen, name, solver.ObjectiveValue() / SCALE


def cpsat_schedule(scn: Scenario, pre: Preprocessed, time_limit: float = 20.0) -> Schedule:
    t0 = time.perf_counter()
    chosen, status, obj = solve_cpsat(pre.options, full_residual(scn),
                                      scn.config.unassigned_penalty, time_limit)
    unassigned = [e.ev_id for e in scn.evs if e.ev_id not in chosen]
    return Schedule("OR-Tools CP-SAT", chosen, unassigned, time.perf_counter() - t0,
                    {"status": status, "objective": obj})
