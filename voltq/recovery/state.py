"""Mutable network state that events act on and recovery strategies read."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..models import Option, Scenario, Session, assign_ports
from .commitments import Commitment, RecoveryWeights


@dataclass
class NetworkState:
    scn: Scenario
    refs: dict[str, float]                       # normalisation refs of the ORIGINAL scenario
    assignments: dict[int, Option]               # ev_id -> current reservation / session
    commitments: dict[int, Commitment]
    t_now: int = 0                               # sessions with start < t_now are in progress / done
    history: dict[int, dict] = field(default_factory=dict)   # ev_id -> {"reschedules","delays","interruptions"}
    sunk_cost_inr: float = 0.0                   # energy already delivered in interrupted sessions
    event_log: list[str] = field(default_factory=list)
    interrupted: set[int] = field(default_factory=set)       # EVs whose in-progress session was cut
    weights: RecoveryWeights = field(default_factory=RecoveryWeights)

    def copy(self) -> "NetworkState":
        return replace(
            self, scn=replace(self.scn, evs=list(self.scn.evs), stations=list(self.scn.stations),
                              prices=list(self.scn.prices), price_tiers=list(self.scn.price_tiers)),
            assignments=dict(self.assignments), commitments=dict(self.commitments),
            history={k: dict(v) for k, v in self.history.items()},
            event_log=list(self.event_log), interrupted=set(self.interrupted))

    def frozen(self) -> dict[int, Option]:
        """Sessions already started: cannot be changed by recovery."""
        return {i: o for i, o in self.assignments.items() if o.start < self.t_now}

    def sessions(self) -> list[Session]:
        return assign_ports(self.assignments, self.scn.stations)

    def hist(self, ev_id: int) -> dict:
        return self.history.setdefault(ev_id, {"reschedules": 0, "delays": 0, "interruptions": 0})
