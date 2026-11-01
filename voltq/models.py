"""Core data models for VoltQ (all units documented per field)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScenarioConfig:
    """Simulation parameters. Every field is configurable; seed makes data reproducible."""

    n_evs: int = 10
    n_stations: int = 4
    ports_min: int = 2
    ports_max: int = 2
    n_slots: int = 12
    slot_minutes: int = 30
    seed: int = 42
    start_hour: float = 19.0          # wall-clock hour at slot 0
    area_km: float = 20.0             # side of square city area
    speed_kmh: float = 40.0           # average driving speed
    efficiency: float = 0.92          # grid->battery charging efficiency
    battery_choices: tuple = (35.0, 50.0, 60.0, 75.0, 90.0)   # kWh
    ev_power_choices: tuple = (11.0, 22.0, 50.0, 100.0)       # kW max accepted
    station_power_choices: tuple = (22.0, 50.0, 100.0)        # kW per port
    restricted_station_prob: float = 0.25   # chance a station closes early
    # tariff (INR/kWh) and tier boundaries (hours of day)
    peak_price: float = 14.0
    normal_price: float = 9.0
    offpeak_price: float = 5.0
    peak_start_hour: float = 17.0
    peak_end_hour: float = 21.0
    offpeak_start_hour: float = 22.0
    offpeak_end_hour: float = 6.0
    price_noise: float = 0.03
    # objective weights (dimensionless, applied to normalized components)
    alpha: float = 1.0                # energy
    beta: float = 1.0                 # waiting
    gamma: float = 1.0                # travel
    wait_inr_per_hour: float = 250.0   # value of waiting time
    travel_inr_per_km: float = 8.0    # cost of driving to a station
    unassigned_penalty: float = 10.0  # objective penalty per unserved EV

    @property
    def slot_hours(self) -> float:
        return self.slot_minutes / 60.0


@dataclass(frozen=True)
class EV:
    ev_id: int
    name: str
    battery_kwh: float
    soc_pct: float
    target_pct: float
    arrival_slot: int        # first slot the EV is available (at its location)
    departure_slot: int      # charging must end at or before this slot boundary
    max_kw: float
    x_km: float
    y_km: float


@dataclass(frozen=True)
class Station:
    station_id: int
    name: str
    ports: int
    power_kw: float
    x_km: float
    y_km: float
    open_slot: int           # first usable slot
    close_slot: int          # charging must end at or before this boundary
    price_factor: float      # multiplier on the base tariff (fast = pricier)


@dataclass
class Scenario:
    config: ScenarioConfig
    evs: list[EV]
    stations: list[Station]
    prices: list[float]      # base tariff INR/kWh per slot
    price_tiers: list[str]   # peak / normal / offpeak per slot


@dataclass(frozen=True)
class Option:
    """One candidate assignment x[v,s,t]: EV v charges at station s from slot `start`."""

    ev_id: int
    station_id: int
    start: int
    duration: int            # slots, contiguous
    energy_kwh: float        # energy drawn from the grid
    energy_cost: float       # INR
    wait_h: float            # idle hours at station before charging starts
    wait_cost: float         # INR
    distance_km: float
    travel_cost: float       # INR
    cost: float = 0.0        # normalized weighted objective contribution

    @property
    def slots(self) -> range:
        return range(self.start, self.start + self.duration)


@dataclass(frozen=True)
class Session:
    ev_id: int
    station_id: int
    port: int                # -1 means no port could be assigned (capacity violation)
    start: int
    duration: int


@dataclass
class Schedule:
    name: str
    assignments: dict[int, Option]
    unassigned: list[int]
    runtime_s: float = 0.0
    meta: dict = field(default_factory=dict)

    def sessions(self, stations: list[Station]) -> list[Session]:
        return assign_ports(self.assignments, stations)


def assign_ports(assignments: dict[int, Option], stations: list[Station]) -> list[Session]:
    """Interval-partition port assignment (optimal for interval graphs).

    If a station's concurrent load exceeds its ports, overflow sessions get port -1.
    """
    out: list[Session] = []
    for st in stations:
        opts = sorted((o for o in assignments.values() if o.station_id == st.station_id),
                      key=lambda o: (o.start, o.ev_id))
        free_at = [0] * st.ports
        for o in opts:
            port = next((p for p in range(st.ports) if free_at[p] <= o.start), -1)
            if port >= 0:
                free_at[port] = o.start + o.duration
            out.append(Session(o.ev_id, st.station_id, port, o.start, o.duration))
    return out
