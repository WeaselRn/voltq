"""Reproducible synthetic scenario generation and CSV/JSON export."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .models import EV, Scenario, ScenarioConfig, Station


def _tier(cfg: ScenarioConfig, hour: float) -> str:
    if cfg.peak_start_hour <= hour < cfg.peak_end_hour:
        return "peak"
    if hour >= cfg.offpeak_start_hour or hour < cfg.offpeak_end_hour:
        return "offpeak"
    return "normal"


def generate_scenario(cfg: ScenarioConfig) -> Scenario:
    """Build a synthetic but realistic scenario; identical for identical cfg (incl. seed)."""
    rng = np.random.default_rng(cfg.seed)
    T, h = cfg.n_slots, cfg.slot_hours

    base = {"peak": cfg.peak_price, "normal": cfg.normal_price, "offpeak": cfg.offpeak_price}
    tiers, prices = [], []
    for t in range(T):
        hour = (cfg.start_hour + (t + 0.5) * h) % 24
        tier = _tier(cfg, hour)
        tiers.append(tier)
        prices.append(round(base[tier] * (1 + rng.normal(0, cfg.price_noise)), 4))

    stations = []
    for i in range(cfg.n_stations):
        power = float(rng.choice(cfg.station_power_choices))
        close = T
        if rng.random() < cfg.restricted_station_prob:
            close = T - int(rng.integers(1, 3))
        stations.append(Station(
            station_id=i, name=f"S{i + 1}",
            ports=int(rng.integers(cfg.ports_min, cfg.ports_max + 1)),
            power_kw=power,
            x_km=round(float(rng.uniform(0, cfg.area_km)), 2),
            y_km=round(float(rng.uniform(0, cfg.area_km)), 2),
            open_slot=0, close_slot=close,
            price_factor=round(1 + 0.003 * (power - min(cfg.station_power_choices)), 4),
        ))

    best_station_kw = max(s.power_kw for s in stations)
    evs = []
    for i in range(cfg.n_evs):
        battery = float(rng.choice(cfg.battery_choices))
        soc = float(rng.integers(10, 51))
        target = float(rng.integers(75, 96))
        max_kw = float(rng.choice(cfg.ev_power_choices))
        arrival = int(rng.integers(0, max(1, T // 2)))
        energy = battery * (target - soc) / 100 / cfg.efficiency
        d_min = math.ceil(energy / (min(max_kw, best_station_kw) * h) - 1e-9)
        departure = min(T, arrival + d_min + int(rng.integers(2, 6)) + 2)
        evs.append(EV(
            ev_id=i, name=f"EV{i + 1:02d}", battery_kwh=battery, soc_pct=soc, target_pct=target,
            arrival_slot=arrival, departure_slot=departure, max_kw=max_kw,
            x_km=round(float(rng.uniform(0, cfg.area_km)), 2),
            y_km=round(float(rng.uniform(0, cfg.area_km)), 2),
        ))
    return Scenario(cfg, evs, stations, prices, tiers)


def scenario_tables(scn: Scenario, distance: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    cfg = scn.config
    ev_df = pd.DataFrame([e.__dict__ for e in scn.evs])
    st_df = pd.DataFrame([s.__dict__ for s in scn.stations])
    hours = [(cfg.start_hour + t * cfg.slot_hours) % 24 for t in range(cfg.n_slots)]
    price_df = pd.DataFrame({
        "slot": range(cfg.n_slots),
        "clock": [f"{int(h):02d}:{int(round((h % 1) * 60)):02d}" for h in hours],
        "tier": scn.price_tiers, "price_eur_kwh": scn.prices,
    })
    tables = {"evs": ev_df, "stations": st_df, "prices": price_df}
    if distance is not None:
        tables["distance_matrix"] = distance
    return tables


def save_scenario(scn: Scenario, out_dir: str | Path, distance: pd.DataFrame | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in scenario_tables(scn, distance).items():
        df.to_csv(out / f"{name}.csv", index=name == "distance_matrix")
    (out / "scenario.json").write_text(json.dumps({
        "config": asdict(scn.config),
        "evs": [e.__dict__ for e in scn.evs],
        "stations": [s.__dict__ for s in scn.stations],
        "prices": scn.prices, "price_tiers": scn.price_tiers,
    }, indent=2))
