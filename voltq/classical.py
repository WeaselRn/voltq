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
                penalty_by_ev: dict[int, float] | None = None,
                power_residual: Callable[[int, int], float] | None = None,
                ) -> tuple[dict[int, Option], str, float]:
    """Max-service min-cost assignment. Returns (assignments, status, objective).

    penalty_by_ev   per-EV unserved penalty (recovery: critical EVs cost far more to drop)
    power_residual  kW still available at (station, slot); enforces the aggregate grid limit
    """
    m = cp_model.CpModel()
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    opts: list[Option] = []
    cover: dict[tuple[int, int], list] = {}
    unassigned_vars = []
    opts_by_var: dict[int, Option] = {}
    obj = []
    for ev_id, lst in options_by_ev.items():
        ev_vars = []
        for o in lst:
            if not fits(o, residual):
                continue
            if power_residual and any(power_residual(o.station_id, t) < o.power_kw - 1e-9 for t in o.slots):
                continue
            v = m.NewBoolVar(f"x_{len(opts)}")
            x[(len(opts), ev_id)] = v
            opts.append(o)
            opts_by_var[id(v)] = o
            ev_vars.append(v)
            obj.append(int(round(o.cost * SCALE)) * v)
            for t in o.slots:
                cover.setdefault((o.station_id, t), []).append(v)
        u = m.NewBoolVar(f"u_{ev_id}")
        m.Add(sum(ev_vars) + u == 1)
        unassigned_vars.append(u)
        pen = (penalty_by_ev or {}).get(ev_id, unassigned_penalty)
        obj.append(int(round(pen * SCALE)) * u)
    for (s, t), vs in cover.items():
        m.Add(sum(vs) <= residual(s, t))
        if power_residual:
            watts = [int(round(opts_by_var[id(v)].power_kw * 100)) for v in vs]
            cap = int(power_residual(s, t) * 100 + 1e-6)
            if sum(watts) > cap:
                m.Add(sum(w * v for w, v in zip(watts, vs)) <= cap)
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
