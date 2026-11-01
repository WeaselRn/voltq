"""Predefined, seeded disruption scenarios (fully offline, deterministic).

Each scenario builds the same initial network (12 EVs, 4 stations, 2 ports each) and returns the
events to trigger. Targets (which station fails, which EV changes its departure) are chosen by rule
from the initial schedule, so the scenario stays meaningful for any seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..models import ScenarioConfig
from ..preprocess import build_options
from ..recovery.commitments import CommitmentType
from ..recovery.events import (DepartureChange, Event, NewArrival, PowerReduction, SessionOverrun,
                               StationOutage, TariffChange)
from ..recovery.state import NetworkState
from .network_simulator import build_initial_state

EVENT_SLOT = 3


def default_config(seed: int = 42, n_evs: int = 12) -> ScenarioConfig:
    return ScenarioConfig(n_evs=n_evs, n_stations=4, ports_min=2, ports_max=2, seed=seed)


def busiest_station(state: NetworkState, slot: int) -> int:
    """Station holding the most sessions that are in progress or still to start at `slot`."""
    load = {s.station_id: 0 for s in state.scn.stations}
    for o in state.assignments.values():
        if o.start + o.duration > slot:
            load[o.station_id] += 1
    return max(load, key=lambda s: (load[s], -s))


def arrival_near(state: NetworkState, station_id: int, slot: int, k: int, seed: int) -> NewArrival:
    rng = np.random.default_rng(seed * 100 + k)
    st = state.scn.stations[station_id]
    return NewArrival(slot, battery_kwh=float(rng.choice([50.0, 60.0, 75.0])),
                      soc_pct=float(rng.integers(10, 30)), target_pct=float(rng.integers(80, 92)),
                      max_kw=float(rng.choice([22.0, 50.0])), x_km=st.x_km, y_km=st.y_km,
                      departure_slot=min(state.scn.config.n_slots, slot + 7 + k))


def tightened_departure(state: NetworkState, slot: int) -> DepartureChange:
    """Pick the protected EV with a future session and pull its departure as early as is still
    physically possible at some station (forces a change, but the EV stays servable)."""
    pre = build_options(state.scn, refs=state.refs, min_start=slot)
    cands = []
    for i, o in state.assignments.items():
        if o.start >= slot and state.commitments[i].ctype in (CommitmentType.CRITICAL,
                                                               CommitmentType.CONFIRMED):
            ends = [x.start + x.duration for x in pre.options[i]]
            if ends and min(ends) < o.start + o.duration:
                cands.append((o.start, i, min(ends)))
    if not cands:                                    # fall back: any future EV, 1 slot earlier
        i, o = max(((i, o) for i, o in state.assignments.items() if o.start >= slot),
                   key=lambda p: p[1].start)
        return DepartureChange(slot, i, o.start + o.duration)
    _, i, earliest_end = max(cands)
    return DepartureChange(slot, i, earliest_end)


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    blurb: str
    build: Callable[[NetworkState, int], list[Event]]


def _failure(state, seed):
    return [StationOutage(EVENT_SLOT, busiest_station(state, EVENT_SLOT))]


def _arrivals(state, seed):
    s = busiest_station(state, EVENT_SLOT)
    return [arrival_near(state, s, EVENT_SLOT, k, seed) for k in range(2)]


def _power(state, seed):
    return [PowerReduction(EVENT_SLOT, busiest_station(state, EVENT_SLOT), 0.6)]


def _departure(state, seed):
    return [tightened_departure(state, EVENT_SLOT)]


def _combined(state, seed):
    out = StationOutage(EVENT_SLOT, busiest_station(state, EVENT_SLOT))
    arr = arrival_near(state, (out.station_id + 1) % len(state.scn.stations), EVENT_SLOT, 0, seed)
    dep = tightened_departure(state, EVENT_SLOT)
    evs: list[Event] = [out, arr]
    if state.assignments.get(dep.ev_id) is None or state.assignments[dep.ev_id].station_id != out.station_id:
        evs.append(dep)
    return evs


def _tariff(state, seed):
    return [TariffChange(EVENT_SLOT, EVENT_SLOT + 2, EVENT_SLOT + 6, 0.5)]


def _overrun(state, seed):
    run = [(o.start, i) for i, o in state.assignments.items()
           if o.start < EVENT_SLOT < o.start + o.duration]
    i = min(run)[1] if run else min(state.assignments, key=lambda k: state.assignments[k].start)
    return [SessionOverrun(EVENT_SLOT, i, 2)]


SCENARIOS: dict[str, Scenario] = {s.key: s for s in [
    Scenario("failure", "1 · Station failure",
             "The busiest station goes down after reservations were confirmed.", _failure),
    Scenario("arrivals", "2 · Unexpected arrivals",
             "Two EVs arrive without a reservation while others are already booked.", _arrivals),
    Scenario("power", "3 · Grid power reduction",
             "The busiest station loses 40% of its available grid power.", _power),
    Scenario("departure", "4 · Departure deadline change",
             "A protected driver leaves much earlier than planned.", _departure),
    Scenario("combined", "5 · Combined network disruption",
             "Station failure + a new arrival + an earlier departure at the same time.", _combined),
    Scenario("tariff", "6 · Tariff change",
             "Electricity halves in price for four slots; can charging shift to profit?", _tariff),
    Scenario("overrun", "7 · Session overrun",
             "A charging EV keeps its port two slots longer than planned.", _overrun),
]}


def build_scenario(key: str, seed: int = 42, solver: str = "CP-SAT", n_evs: int = 12,
                   **kw) -> tuple[NetworkState, list[Event]]:
    state = build_initial_state(default_config(seed, n_evs), solver, **kw)
    state.t_now = 0
    return state, SCENARIOS[key].build(state, seed)
