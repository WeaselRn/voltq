"""Affected-EV detection.

An event names the EVs it touches directly (`EventEffect.directly_affected`). On top of that the
detector re-checks every current reservation against the NEW network using the independent
validator, so it also finds EVs the event did not name:

* invalidated  - the reservation itself breaks a rule now (station closed, deadline moved, ...)
* conflicts    - the reservation is fine alone but shares an over-full (station, slot) cell
                 (ports or grid kW) with others after the event
* unserved     - not placed before; freed capacity may let them in

Sessions that already started (start < t_now) are frozen and can never be chosen as victims.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..evaluate import evaluate_sessions
from ..models import Option, Session, grid_limit_kw
from .state import NetworkState


@dataclass
class AffectedReport:
    direct: set[int] = field(default_factory=set)
    invalidated: dict[int, list[str]] = field(default_factory=dict)
    conflicts: set[int] = field(default_factory=set)
    unserved: set[int] = field(default_factory=set)
    frozen: set[int] = field(default_factory=set)
    unresolvable: list[str] = field(default_factory=list)   # overloads caused by frozen sessions only
    interrupted: set[int] = field(default_factory=set)      # in-progress session cut by the event

    @property
    def affected(self) -> set[int]:
        return (self.direct | set(self.invalidated) | self.conflicts | self.unserved | self.interrupted) - self.frozen

    def reasons(self, ev_id: int) -> list[str]:
        r = []
        if ev_id in self.direct:
            r.append("named by the event")
        if ev_id in self.invalidated:
            r.append("reservation no longer valid: " + "; ".join(self.invalidated[ev_id][:2]))
        if ev_id in self.conflicts:
            r.append("shares an over-capacity slot")
        if ev_id in self.interrupted:
            r.append("session interrupted mid-charge")
        elif ev_id in self.unserved:
            r.append("was unserved (may fit now)")
        return r


def invalid_reservations(state: NetworkState) -> dict[int, list[str]]:
    """EVs whose reservation breaks a hard rule on its own in the post-event network."""
    out: dict[int, list[str]] = {}
    for ev_id, o in state.assignments.items():
        m = evaluate_sessions(state.scn, [Session(ev_id, o.station_id, 0, o.start, o.duration)],
                              state.refs)
        if not m["valid_partial"]:
            out[ev_id] = m["violations"]
    return out


def overloaded_cells(state: NetworkState, assignments: dict[int, Option]
                     ) -> dict[tuple[int, int], list[int]]:
    """(station, slot) cells where ports or grid kW are exceeded -> EVs covering the cell."""
    ports = {s.station_id: s for s in state.scn.stations}
    cover: dict[tuple[int, int], list[int]] = {}
    for ev_id, o in assignments.items():
        for t in o.slots:
            cover.setdefault((o.station_id, t), []).append(ev_id)
    bad = {}
    for (s, t), ids in cover.items():
        st = ports[s]
        kw = sum(assignments[i].power_kw for i in ids)
        if len(ids) > st.ports or kw > grid_limit_kw(st, t) + 1e-9:
            bad[(s, t)] = ids
    return bad


def identify_affected(state: NetworkState, direct: set[int]) -> AffectedReport:
    frozen = set(state.frozen())
    inv = {i: v for i, v in invalid_reservations(state).items()}
    rep = AffectedReport(direct=set(direct) - frozen, invalidated={i: v for i, v in inv.items()
                                                                  if i not in frozen},
                         frozen=frozen)
    valid = {i: o for i, o in state.assignments.items() if i not in inv}
    for cell, ids in overloaded_cells(state, valid).items():
        movable = [i for i in ids if i not in frozen]
        if movable:
            rep.conflicts.update(movable)
        else:
            rep.unresolvable.append(f"frozen sessions overload station {cell[0]} slot {cell[1]}")
    rep.interrupted = set(state.interrupted)
    rep.unserved = {e.ev_id for e in state.scn.evs if e.ev_id not in state.assignments
                    and e.ev_id not in frozen and e.ev_id not in rep.interrupted}
    return rep


def static_schedule(state: NetworkState) -> dict[int, Option]:
    """'No recovery': keep every reservation that survives the event; sessions that cannot run
    (invalid, or squeezed out of an over-full cell, latest start first) simply fail."""
    frozen = set(state.frozen())
    keep = {i: o for i, o in state.assignments.items() if i not in invalid_reservations(state)}
    while True:
        bad = overloaded_cells(state, keep)
        victims = [i for ids in bad.values() for i in ids if i not in frozen]
        if not victims:
            return keep
        drop = max(victims, key=lambda i: (keep[i].start, i))
        keep.pop(drop)
