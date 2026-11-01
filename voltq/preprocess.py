"""Classical preprocessing: energy, duration, distance, feasible (EV, station, start) options.

Assumptions (also in README):
* Charging power at a port is constant = min(EV max kW, station kW) (no taper curve).
* Grid energy = battery energy / efficiency. The final slot may be partially used.
* Session = one contiguous block on one port (no pre-emption).
* Travel: straight-line distance, constant speed. The EV can start charging at
  station s no earlier than arrival_slot + ceil(travel_time / slot_len).
* Waiting = hours between arriving at the station and starting to charge.
* Station capacity = number of ports; ports are interchangeable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pandas as pd

from .models import EV, Option, Scenario, Station


def required_energy_kwh(ev: EV, efficiency: float) -> float:
    """Grid energy needed to go from soc to target."""
    return ev.battery_kwh * max(ev.target_pct - ev.soc_pct, 0.0) / 100.0 / efficiency


def effective_power_kw(ev: EV, st: Station) -> float:
    return min(ev.max_kw, st.power_kw)


def charge_duration_slots(energy_kwh: float, power_kw: float, slot_hours: float) -> int:
    if energy_kwh <= 0:
        return 0
    return math.ceil(energy_kwh / (power_kw * slot_hours) - 1e-9)


def distance_km(ev: EV, st: Station) -> float:
    return math.hypot(ev.x_km - st.x_km, ev.y_km - st.y_km)


def travel_slots(dist_km: float, speed_kmh: float, slot_hours: float) -> int:
    return math.ceil(dist_km / speed_kmh / slot_hours - 1e-9)


def energy_profile(energy_kwh: float, power_kw: float, slot_hours: float, duration: int) -> list[float]:
    """kWh drawn in each slot of a session."""
    out, remaining = [], energy_kwh
    for _ in range(duration):
        e = min(power_kw * slot_hours, remaining)
        out.append(e)
        remaining -= e
    return out


def distance_matrix(scn: Scenario) -> pd.DataFrame:
    return pd.DataFrame(
        [[round(distance_km(e, s), 3) for s in scn.stations] for e in scn.evs],
        index=[e.name for e in scn.evs], columns=[s.name for s in scn.stations])


@dataclass
class Preprocessed:
    options: dict[int, list[Option]]     # ev_id -> feasible options (cost filled in)
    refs: dict[str, float]               # normalization references (EUR)
    infeasible_evs: list[int]            # EVs with no feasible option at all
    full_size: int                       # naive |V|*|S|*|T| variable count
    reduced_size: int                    # number of feasible options kept

    def all_options(self) -> list[Option]:
        return [o for opts in self.options.values() for o in opts]


def build_options(scn: Scenario) -> Preprocessed:
    cfg, h = scn.config, scn.config.slot_hours
    raw: dict[int, list[Option]] = {}
    for ev in scn.evs:
        energy = required_energy_kwh(ev, cfg.efficiency)
        opts = []
        for st in scn.stations:
            p = effective_power_kw(ev, st)
            dur = charge_duration_slots(energy, p, h)
            dist = distance_km(ev, st)
            earliest = max(ev.arrival_slot + travel_slots(dist, cfg.speed_kmh, h), st.open_slot)
            last_end = min(ev.departure_slot, st.close_slot, cfg.n_slots)
            if dur == 0:
                continue
            prof = energy_profile(energy, p, h, dur)
            for t in range(earliest, last_end - dur + 1):
                e_cost = sum(prof[k] * scn.prices[t + k] * st.price_factor for k in range(dur))
                arrive_slot = ev.arrival_slot + travel_slots(dist, cfg.speed_kmh, h)
                wait_h = (t - arrive_slot) * h
                opts.append(Option(
                    ev_id=ev.ev_id, station_id=st.station_id, start=t, duration=dur,
                    energy_kwh=energy, energy_cost=e_cost, wait_h=wait_h,
                    wait_cost=wait_h * cfg.wait_eur_per_hour,
                    distance_km=dist, travel_cost=dist * cfg.travel_eur_per_km))
        raw[ev.ev_id] = opts

    flat = [o for v in raw.values() for o in v]

    def mean_pos(vals: list[float]) -> float:
        pos = [x for x in vals if x > 1e-12]
        return sum(pos) / len(pos) if pos else 1.0

    refs = {
        "energy": mean_pos([o.energy_cost for o in flat]),
        "waiting": mean_pos([o.wait_cost for o in flat]),
        "travel": mean_pos([o.travel_cost for o in flat]),
    }
    options = {
        ev_id: [replace(o, cost=cfg.alpha * o.energy_cost / refs["energy"]
                        + cfg.beta * o.wait_cost / refs["waiting"]
                        + cfg.gamma * o.travel_cost / refs["travel"]) for o in opts]
        for ev_id, opts in raw.items()
    }
    return Preprocessed(
        options=options, refs=refs,
        infeasible_evs=[i for i, v in options.items() if not v],
        full_size=len(scn.evs) * len(scn.stations) * cfg.n_slots,
        reduced_size=len(flat))
