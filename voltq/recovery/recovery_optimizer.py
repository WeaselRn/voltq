"""Recovery orchestration: affected-set detection -> local re-optimisation with staged expansion.

    stage 0  directly affected + invalidated + conflicting (+ previously unserved) EVs
    stage 1  + FLEXIBLE / UNCONFIRMED EVs that hold capacity at stations the unserved EVs could use
    stage 2  + all FLEXIBLE / UNCONFIRMED EVs
    stage 3  + CONFIRMED EVs (only if protect_confirmed != "strict" or a CRITICAL EV is still unserved)

The set only grows while some EV in it is still unserved. `scope="full"` frees every EV whose
session has not started (comparison baseline: maximum freedom, minimum schedule stability).
Frozen (already started) sessions are never touched.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..hybrid import HybridConfig
from ..models import Option
from ..qaoa import QAOAConfig
from .commitments import CommitmentType
from .detection import AffectedReport, identify_affected, static_schedule
from .problem import RecoveryProblem, build_problem
from .solvers import cpsat_recover, greedy_recover, hybrid_recover
from .state import NetworkState

STRATEGIES = ("Static (no recovery)", "Greedy recovery", "CP-SAT recovery",
              "Hybrid QAOA recovery", "CP-SAT full-network")


@dataclass
class RecoveryResult:
    strategy: str
    assignments: dict[int, Option]            # complete post-recovery schedule
    free_ids: list[int]
    runtime_s: float
    stages: int = 0
    meta: dict = field(default_factory=dict)
    logs: list = field(default_factory=list)  # hybrid: per-batch QAOA logs


def _expand(state: NetworkState, free: set[int], unserved: set[int], stage: int,
            frozen: set[int], fixed: dict[int, Option], protect: str) -> set[int]:
    movable = {i: o for i, o in fixed.items() if i not in frozen}
    ct = lambda i: state.commitments[i].ctype
    soft = {CommitmentType.FLEXIBLE, CommitmentType.UNCONFIRMED}
    if stage == 1:
        stations = set()
        for u in unserved:
            ev = state.scn.evs[u]
            stations |= {s.station_id for s in state.scn.stations
                         if s.close_slot > ev.arrival_slot and s.open_slot < ev.departure_slot}
        return free | {i for i, o in movable.items() if ct(i) in soft and o.station_id in stations}
    if stage == 2:
        return free | {i for i in movable if ct(i) in soft}
    crit = any(ct(u) == CommitmentType.CRITICAL for u in unserved)
    if protect != "strict" or crit:
        return free | {i for i in movable if ct(i) == CommitmentType.CONFIRMED}
    return free


def recover(state: NetworkState, direct: set[int], strategy: str, *, scope: str = "local",
            qcfg: QAOAConfig | None = None, hcfg: HybridConfig | None = None,
            time_limit: float = 20.0, report: AffectedReport | None = None) -> RecoveryResult:
    """Run one strategy on the post-event `state`. `state` is not modified."""
    t0 = time.perf_counter()
    if strategy.startswith("Static"):
        return RecoveryResult(strategy, static_schedule(state), [], time.perf_counter() - t0)

    report = report or identify_affected(state, direct)
    frozen = set(state.frozen())
    w = state.weights
    full = scope == "full" or strategy == "CP-SAT full-network"
    movable_all = {e.ev_id for e in state.scn.evs if e.ev_id not in frozen}
    free = set(movable_all) if full else set(report.affected)
    meta: dict = {"affected_initial": sorted(report.affected), "scope": "full" if full else "local"}
    logs: list = []
    stage = 0
    while True:
        fixed = {i: o for i, o in state.assignments.items()
                 if i in frozen or (i not in free and i not in report.invalidated)}
        prob = build_problem(state, sorted(free), fixed, w)
        if not free:
            got = {}
        elif strategy.startswith("Greedy"):
            got = greedy_recover(prob)
        elif strategy.startswith("CP-SAT"):
            got, info = cpsat_recover(prob, time_limit)
            meta.update(info)
        elif strategy.startswith("Hybrid"):
            got, logs, info = hybrid_recover(prob, qcfg or QAOAConfig(maxiter=40, shots=256), hcfg)
            meta.update(info)
        else:
            raise ValueError(f"unknown strategy {strategy}")
        servable = {i for i in free if prob.options[i]}
        unserved = servable - set(got)
        if not unserved or full or stage >= 3:
            break
        stage += 1
        bigger = _expand(state, free, unserved, stage, frozen, fixed, w.protect_confirmed)
        while bigger == free and stage < 3:          # skip stages that add nothing
            stage += 1
            bigger = _expand(state, free, unserved, stage, frozen, fixed, w.protect_confirmed)
        if bigger == free:
            break
        free = bigger
    meta["affected_final"] = sorted(free)
    meta["unserved_options_exist"] = sorted(unserved)
    meta["unavoidable_no_option"] = sorted(i for i in free if not prob.options[i])
    return RecoveryResult(strategy, {**fixed, **got}, sorted(free), time.perf_counter() - t0,
                          stage, meta, logs)
