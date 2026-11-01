"""Disruption events with one common interface.

Every event is a frozen dataclass with `slot` (when it happens) and `apply(state)`, which returns an
`EventEffect` holding a NEW state plus the EVs the event touches directly. A new disruption type is
added by subclassing `Event` and decorating with `@register`; the detection, recovery and metrics
code never needs to change because it only looks at the resulting state (see `detection.py`).

The structures mirror OCPP-style messages (StatusNotification, ChangeAvailability, SetChargingProfile,
RemoteStopTransaction) but only simulated telemetry is used: no charger is contacted.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

from ..models import EV, Option, grid_limit_kw
from ..preprocess import energy_profile
from .commitments import PRIORITY, Commitment, CommitmentType
from .state import NetworkState

EVENT_TYPES: dict[str, type["Event"]] = {}


def register(cls):
    EVENT_TYPES[cls.kind] = cls
    return cls


@dataclass
class EventEffect:
    state: NetworkState
    directly_affected: set[int] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Event:
    kind: ClassVar[str] = "event"
    slot: int

    def apply(self, state: NetworkState) -> EventEffect:       # pragma: no cover - interface
        raise NotImplementedError

    def describe(self, state=None) -> str:                      # pragma: no cover - interface
        return self.kind


def _end(o: Option) -> int:
    return o.start + o.duration


def in_progress(state: NetworkState, slot: int) -> dict[int, Option]:
    return {i: o for i, o in state.assignments.items() if o.start < slot < _end(o)}


def interrupt_session(state: NetworkState, ev_id: int, slot: int) -> float:
    """Cut an in-progress session at `slot`. The EV keeps the energy already delivered, stays at the
    station's location and re-enters the pool with a new (higher) SoC. Returns delivered grid kWh."""
    cfg = state.scn.config
    o = state.assignments.pop(ev_id)
    prof = energy_profile(o.energy_kwh, o.power_kw, cfg.slot_hours, o.duration)
    done = sum(prof[:slot - o.start])
    st = state.scn.stations[o.station_id]
    ev = state.scn.evs[ev_id]
    state.scn.evs[ev_id] = replace(
        ev, soc_pct=ev.soc_pct + done * cfg.efficiency / ev.battery_kwh * 100.0,
        arrival_slot=slot, x_km=st.x_km, y_km=st.y_km)
    state.sunk_cost_inr += sum(prof[k] * state.scn.prices[o.start + k] * st.price_factor
                               for k in range(slot - o.start))
    state.interrupted.add(ev_id)
    state.hist(ev_id)["interruptions"] += 1
    return done


@register
@dataclass(frozen=True)
class StationOutage(Event):
    kind: ClassVar[str] = "station_outage"
    station_id: int = 0

    def apply(self, state):
        s, notes, direct = state.copy(), [], set()
        st = s.scn.stations[self.station_id]
        s.scn.stations[self.station_id] = replace(st, close_slot=min(st.close_slot, self.slot))
        for ev_id, o in list(s.assignments.items()):
            if o.station_id != self.station_id:
                continue
            if o.start < self.slot < _end(o):
                done = interrupt_session(s, ev_id, self.slot)
                notes.append(f"{s.scn.evs[ev_id].name} interrupted after {done:.1f} kWh")
                direct.add(ev_id)
            elif o.start >= self.slot:
                direct.add(ev_id)
        s.t_now = max(s.t_now, self.slot)
        s.event_log.append(self.describe(s))
        return EventEffect(s, direct, notes)

    def describe(self, state=None) -> str:
        name = state.scn.stations[self.station_id].name if state else f"station {self.station_id}"
        return f"[slot {self.slot}] Station outage: {name} unavailable from slot {self.slot}"


@register
@dataclass(frozen=True)
class NewArrival(Event):
    kind: ClassVar[str] = "new_arrival"
    battery_kwh: float = 60.0
    soc_pct: float = 20.0
    target_pct: float = 85.0
    max_kw: float = 50.0
    x_km: float = 10.0
    y_km: float = 10.0
    departure_slot: int = 12
    ctype: CommitmentType = CommitmentType.UNCONFIRMED

    def apply(self, state):
        s = state.copy()
        ev_id = len(s.scn.evs)
        ev = EV(ev_id, f"EV{ev_id + 1:02d}", self.battery_kwh, self.soc_pct, self.target_pct,
                self.slot, min(self.departure_slot, s.scn.config.n_slots), self.max_kw,
                self.x_km, self.y_km)
        s.scn.evs.append(ev)
        s.commitments[ev_id] = Commitment(ev_id, self.ctype, PRIORITY[self.ctype], 0)
        s.t_now = max(s.t_now, self.slot)
        s.event_log.append(self.describe(s))
        return EventEffect(s, {ev_id}, [f"{ev.name} arrived (SoC {ev.soc_pct:.0f}% -> {ev.target_pct:.0f}%)"])

    def describe(self, state=None) -> str:
        return (f"[slot {self.slot}] New EV arrives: {self.battery_kwh:.0f} kWh, SoC {self.soc_pct:.0f}% "
                f"-> {self.target_pct:.0f}%, leaves at slot {self.departure_slot}")


@register
@dataclass(frozen=True)
class DepartureChange(Event):
    kind: ClassVar[str] = "departure_change"
    ev_id: int = 0
    new_departure: int = 0

    def apply(self, state):
        s = state.copy()
        ev = s.scn.evs[self.ev_id]
        o = s.assignments.get(self.ev_id)
        if o is not None and o.start < self.slot and _end(o) > self.new_departure:
            raise ValueError(f"{ev.name} is already charging past its new departure slot")
        s.scn.evs[self.ev_id] = replace(ev, departure_slot=min(self.new_departure, s.scn.config.n_slots))
        s.t_now = max(s.t_now, self.slot)
        s.event_log.append(self.describe(s))
        direct = {self.ev_id} if (o is None or o.start >= self.slot) else set()
        return EventEffect(s, direct, [f"{ev.name} departure {ev.departure_slot} -> {self.new_departure}"])

    def describe(self, state=None) -> str:
        name = state.scn.evs[self.ev_id].name if state else f"EV {self.ev_id}"
        return f"[slot {self.slot}] {name} changes departure to slot {self.new_departure}"


@register
@dataclass(frozen=True)
class PowerReduction(Event):
    """Aggregate power cap of a station drops to `remaining_fraction` of its current limit."""
    kind: ClassVar[str] = "power_reduction"
    station_id: int = 0
    remaining_fraction: float = 0.6

    def apply(self, state):
        s, notes, direct = state.copy(), [], set()
        st = s.scn.stations[self.station_id]
        new_cap = grid_limit_kw(st, self.slot) * self.remaining_fraction
        s.scn.stations[self.station_id] = replace(st, grid_kw=new_cap, grid_from=self.slot)
        running = sorted(((i, o) for i, o in in_progress(s, self.slot).items()
                          if o.station_id == self.station_id), key=lambda p: (-p[1].start, -p[0]))
        load = sum(o.power_kw for _, o in running)
        for ev_id, o in running:           # cut the most recently started sessions first
            if load <= new_cap + 1e-9:
                break
            interrupt_session(s, ev_id, self.slot)
            load -= o.power_kw
            direct.add(ev_id)
            notes.append(f"{s.scn.evs[ev_id].name} paused by the power cap")
        s.t_now = max(s.t_now, self.slot)
        s.event_log.append(self.describe(s))
        return EventEffect(s, direct, notes)

    def describe(self, state=None) -> str:
        name = state.scn.stations[self.station_id].name if state else f"station {self.station_id}"
        return (f"[slot {self.slot}] Power reduction at {name}: "
                f"{(1 - self.remaining_fraction) * 100:.0f}% less grid power")


@register
@dataclass(frozen=True)
class TariffChange(Event):
    kind: ClassVar[str] = "tariff_change"
    from_slot: int = 0
    to_slot: int = 12
    multiplier: float = 1.0

    def apply(self, state):
        s = state.copy()
        T = s.scn.config.n_slots
        for t in range(max(self.from_slot, 0), min(self.to_slot, T)):
            s.scn.prices[t] = round(s.scn.prices[t] * self.multiplier, 4)
        s.t_now = max(s.t_now, self.slot)
        frozen = set(s.frozen())
        direct = {e.ev_id for e in s.scn.evs if e.ev_id not in frozen
                  and e.arrival_slot < self.to_slot and e.departure_slot > self.from_slot}
        s.event_log.append(self.describe(s))
        return EventEffect(s, direct, [f"{len(direct)} EVs could shift to cheaper slots"])

    def describe(self, state=None) -> str:
        return (f"[slot {self.slot}] Tariff x{self.multiplier:.2f} for slots "
                f"{self.from_slot}-{self.to_slot - 1}")


@register
@dataclass(frozen=True)
class SessionOverrun(Event):
    kind: ClassVar[str] = "session_overrun"
    ev_id: int = 0
    extra_slots: int = 1

    def apply(self, state):
        s = state.copy()
        o = s.assignments.get(self.ev_id)
        if o is None or not (o.start < self.slot < _end(o)):
            raise ValueError("overrun needs a session that is in progress at the event slot")
        T = s.scn.config.n_slots
        st = s.scn.stations[o.station_id]
        new_end = min(_end(o) + self.extra_slots, T, st.close_slot)
        s.assignments[self.ev_id] = replace(o, duration=new_end - o.start)
        ev = s.scn.evs[self.ev_id]
        s.scn.evs[self.ev_id] = replace(ev, departure_slot=max(ev.departure_slot, new_end))
        s.t_now = max(s.t_now, self.slot)
        s.event_log.append(self.describe(s))
        return EventEffect(s, set(), [f"{ev.name} keeps its port until slot {new_end}"])

    def describe(self, state=None) -> str:
        name = state.scn.evs[self.ev_id].name if state else f"EV {self.ev_id}"
        return f"[slot {self.slot}] {name} overruns its session by {self.extra_slots} slot(s)"


def apply_events(state: NetworkState, events: list[Event]) -> EventEffect:
    """Apply several events in order (same recovery round)."""
    cur, direct, notes = state, set(), []
    for ev in events:
        eff = ev.apply(cur)
        cur = eff.state
        direct |= eff.directly_affected
        notes += eff.notes
    return EventEffect(cur, direct, notes)
