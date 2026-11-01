"""Reservation commitments and the recovery objective weights.

Recovery objective (per optimisation, all terms in the same normalised units as the base cost):

    C_recovery = C_operational                       energy + waiting + travel   (existing objective)
               + lam_reassign  * [reservation moved]
               + lam_commit[c] * [reservation moved]   c = commitment type of the EV
               + P_deadline[c] * [EV not served]       missed departure / unserved energy
               + lam_fair * disadvantage_v * wait_v    optional, see `fairness`

Every term is linear in the option variables x[v,s,t], so the SAME coefficients drop into the QUBO
(linear diagonal), CP-SAT (objective) and Greedy (sort key) without a new constraint type.
Critical EVs get a very large P_deadline: a hard requirement wherever physically feasible, and an
explicitly reported violation when it is not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..models import Option, Scenario
from ..preprocess import Preprocessed


class CommitmentType(str, Enum):
    CONFIRMED = "confirmed"        # existing reservation, protected
    FLEXIBLE = "flexible"          # may be moved when the network is constrained
    UNCONFIRMED = "unconfirmed"    # no reservation yet (new arrival)
    CRITICAL = "critical"          # strict departure / energy requirement


PRIORITY = {CommitmentType.CRITICAL: 3, CommitmentType.CONFIRMED: 2,
            CommitmentType.FLEXIBLE: 1, CommitmentType.UNCONFIRMED: 0}


@dataclass(frozen=True)
class Commitment:
    ev_id: int
    ctype: CommitmentType
    priority: int
    flexibility_slots: int         # departure window minus shortest charge (slack)


@dataclass
class RecoveryWeights:
    lam_reassign: float = 0.6
    lam_commit: dict = field(default_factory=lambda: {
        CommitmentType.CONFIRMED: 2.0, CommitmentType.CRITICAL: 3.0,
        CommitmentType.FLEXIBLE: 0.0, CommitmentType.UNCONFIRMED: 0.0})
    deadline_penalty: dict = field(default_factory=lambda: {
        CommitmentType.CRITICAL: 1000.0, CommitmentType.CONFIRMED: 15.0,
        CommitmentType.FLEXIBLE: 10.0, CommitmentType.UNCONFIRMED: 10.0})
    fairness: float = 0.0                 # lam_fair; 0 = off
    protect_confirmed: str = "soft"       # strict: valid confirmed EVs are only touched to save a CRITICAL EV
                                          # soft:   as a last stage, expansion may move them (penalised)
                                          # off:    no protection (lam_reassign = lam_commit = 0)

    def effective(self) -> "RecoveryWeights":
        if self.protect_confirmed != "off":
            return self
        z = {k: 0.0 for k in self.lam_commit}
        return RecoveryWeights(0.0, z, self.deadline_penalty, self.fairness, "off")


def min_duration(pre: Preprocessed, ev_id: int) -> int | None:
    opts = pre.options.get(ev_id) or []
    return min((o.duration for o in opts), default=None)


def assign_commitments(scn: Scenario, pre: Preprocessed, assignments: dict[int, Option],
                       seed: int = 0, critical_frac: float = 0.2,
                       flexible_frac: float = 0.3) -> dict[int, Commitment]:
    """Deterministic commitment mix: the tightest-window EVs are CRITICAL, the rest are
    CONFIRMED or FLEXIBLE (seeded). EVs without an initial reservation are UNCONFIRMED."""
    rng = np.random.default_rng(seed)
    slack: dict[int, int] = {}
    for ev in scn.evs:
        d = min_duration(pre, ev.ev_id)
        slack[ev.ev_id] = (ev.departure_slot - ev.arrival_slot - d) if d is not None else 0
    served = [e.ev_id for e in scn.evs if e.ev_id in assignments]
    n_crit = max(1, math.ceil(critical_frac * len(served))) if served else 0
    tight = set(sorted(served, key=lambda i: (slack[i], i))[:n_crit])
    out: dict[int, Commitment] = {}
    for ev in scn.evs:
        i = ev.ev_id
        if i not in assignments:
            c = CommitmentType.UNCONFIRMED
        elif i in tight:
            c = CommitmentType.CRITICAL
        else:
            c = CommitmentType.FLEXIBLE if rng.random() < flexible_frac else CommitmentType.CONFIRMED
        out[i] = Commitment(i, c, PRIORITY[c], slack[i])
    return out
