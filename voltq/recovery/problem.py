"""Recovery problem definition shared by every strategy (identical inputs for Greedy / CP-SAT / QAOA)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..models import Option, Scenario
from ..preprocess import build_options
from .commitments import CommitmentType, RecoveryWeights
from .state import NetworkState


@dataclass
class RecoveryProblem:
    scn: Scenario
    t_now: int
    fixed: dict[int, Option]                 # frozen + untouched reservations (hard constraints)
    free_ids: list[int]                      # EVs the solver may (re)place
    options: dict[int, list[Option]]         # recovery-augmented cost, only starts >= t_now
    penalty: dict[int, float]                # unserved penalty per free EV
    prev: dict[int, Option]                  # previous reservation of each free EV (if any)
    ctype: dict[int, CommitmentType]
    weights: RecoveryWeights
    priority: dict[int, int] = field(default_factory=dict)

    def cost(self, ev_id: int, o: Option | None) -> float:
        return self.penalty[ev_id] if o is None else o.cost


def option_key(o: Option) -> tuple[int, int, int]:
    return (o.ev_id, o.station_id, o.start)


def augment(o: Option, prev: Option | None, ctype: CommitmentType, w: RecoveryWeights,
            disadvantage: int, wait_ref: float) -> Option:
    """Add reassignment / commitment / fairness terms to the base cost (all linear in x)."""
    extra = 0.0
    if prev is not None and (prev.station_id, prev.start) != (o.station_id, o.start):
        extra += w.lam_reassign + w.lam_commit[ctype]
    if w.fairness > 0 and disadvantage > 0:
        extra += w.fairness * disadvantage * o.wait_cost / wait_ref
    return replace(o, cost=o.cost + extra)


def build_problem(state: NetworkState, free_ids: list[int], fixed: dict[int, Option],
                  weights: RecoveryWeights | None = None) -> RecoveryProblem:
    w = (weights or state.weights).effective()
    free = sorted(free_ids)
    pre = build_options(state.scn, refs=state.refs, min_start=state.t_now, ev_ids=set(free))
    prev = {i: state.assignments[i] for i in free if i in state.assignments}
    ctype = {i: state.commitments[i].ctype for i in free}
    options = {}
    for i in free:
        h = state.history.get(i, {})
        dis = h.get("reschedules", 0) + h.get("delays", 0) + h.get("interruptions", 0)
        options[i] = [augment(o, prev.get(i), ctype[i], w, dis, state.refs["waiting"])
                      for o in pre.options[i]]
    return RecoveryProblem(
        scn=state.scn, t_now=state.t_now, fixed=dict(fixed), free_ids=free, options=options,
        penalty={i: w.deadline_penalty[ctype[i]] for i in free}, prev=prev, ctype=ctype, weights=w,
        priority={i: state.commitments[i].priority for i in free})


def network_objective(state: NetworkState, assignments: dict[int, Option],
                      weights: RecoveryWeights | None = None) -> dict:
    """Recovery objective of a FINAL schedule, recomputed from the schedule itself.

    Covers every EV whose session has not started (frozen sessions are identical in all
    strategies). Same formula the solvers minimise, so strategies are comparable."""
    w = (weights or state.weights).effective()
    frozen = set(state.frozen())
    movable = [e.ev_id for e in state.scn.evs if e.ev_id not in frozen]
    pre = build_options(state.scn, refs=state.refs, min_start=state.t_now, ev_ids=set(movable))
    ctype = {i: state.commitments[i].ctype for i in movable}
    operational = reassign = commit = missed = 0.0
    for i in movable:
        o = assignments.get(i)
        if o is None:
            missed += w.deadline_penalty[ctype[i]]
            continue
        match = next((x for x in pre.options[i]
                      if x.station_id == o.station_id and x.start == o.start), None)
        if match is None:                       # session not valid in the new network
            missed += w.deadline_penalty[ctype[i]]
            continue
        operational += match.cost
        prev = state.assignments.get(i)
        if prev is not None and (prev.station_id, prev.start) != (o.station_id, o.start):
            reassign += w.lam_reassign
            commit += w.lam_commit[ctype[i]]
    return {"operational": operational, "reassignment": reassign, "commitment": commit,
            "missed_deadline": missed, "total": operational + reassign + commit + missed}
