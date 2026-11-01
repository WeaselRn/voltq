"""Independent schedule validator and metrics.

Deliberately re-derives everything (distance, travel delay, delivered energy, costs) from the
raw EV/Station tables and the list of Sessions, without using the solvers' Option objects.
"""
from __future__ import annotations

import math

import numpy as np

from .models import Scenario, Session


def evaluate_sessions(scn: Scenario, sessions: list[Session], refs: dict[str, float],
                      runtime_s: float = 0.0, tol: float = 1e-6) -> dict:
    cfg = scn.config
    h, T = cfg.slot_hours, cfg.n_slots
    evs = {e.ev_id: e for e in scn.evs}
    sts = {s.station_id: s for s in scn.stations}
    violations: list[str] = []

    seen: dict[int, int] = {}
    for s in sessions:
        seen[s.ev_id] = seen.get(s.ev_id, 0) + 1
    for ev_id, c in seen.items():
        if c > 1:
            violations.append(f"EV {ev_id} has {c} simultaneous/multiple assignments")

    occupancy = np.zeros((len(scn.stations), T), dtype=int)
    port_busy: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    e_cost = w_h = dist_total = 0.0
    served = 0
    cap_viol = 0
    for s in sessions:
        ev, st = evs[s.ev_id], sts[s.station_id]
        d = math.hypot(ev.x_km - st.x_km, ev.y_km - st.y_km)
        arrive = ev.arrival_slot + math.ceil(d / cfg.speed_kmh / h - 1e-9)
        end = s.start + s.duration
        if s.start < arrive:
            violations.append(f"EV {s.ev_id} starts slot {s.start} before reaching station (slot {arrive})")
        if s.start < st.open_slot or end > st.close_slot:
            violations.append(f"EV {s.ev_id} outside operating hours of {st.name}")
        if end > ev.departure_slot:
            violations.append(f"EV {s.ev_id} ends slot {end} after departure {ev.departure_slot}")
        if s.start < 0 or end > T:
            violations.append(f"EV {s.ev_id} outside horizon")
            continue
        # simulate energy delivery slot by slot
        power = min(ev.max_kw, st.power_kw)
        need = ev.battery_kwh * (ev.target_pct - ev.soc_pct) / 100 / cfg.efficiency
        delivered, cost = 0.0, 0.0
        for t in range(s.start, end):
            e = min(power * h, need - delivered)
            delivered += e
            cost += e * scn.prices[t] * st.price_factor
        if delivered < need - tol:
            violations.append(f"EV {s.ev_id} gets {delivered:.2f} of {need:.2f} kWh")
        else:
            served += 1
        e_cost += cost
        w_h += max(s.start - arrive, 0) * h
        dist_total += d
        for t in range(s.start, end):
            occupancy[s.station_id, t] += 1
        if s.port < 0 or s.port >= st.ports:
            cap_viol += 1
            violations.append(f"EV {s.ev_id} has no valid port at {st.name}")
        else:
            port_busy.setdefault((s.station_id, s.port), []).append((s.start, end, s.ev_id))

    for (sid, p), ivs in port_busy.items():
        ivs.sort()
        for a, b in zip(ivs, ivs[1:]):
            if b[0] < a[1]:
                cap_viol += 1
                violations.append(f"port overlap at {sts[sid].name} port {p}: EV {a[2]} & EV {b[2]}")
    for st in scn.stations:
        over = int(np.maximum(occupancy[st.station_id] - st.ports, 0).sum())
        if over:
            cap_viol += over
            violations.append(f"{st.name} over capacity in {over} slot-units")

    n = len(scn.evs)
    unassigned = n - len({s.ev_id for s in sessions})
    wait_cost = w_h * cfg.wait_inr_per_hour
    travel_cost = dist_total * cfg.travel_inr_per_km
    objective = (cfg.alpha * e_cost / refs["energy"] + cfg.beta * wait_cost / refs["waiting"]
                 + cfg.gamma * travel_cost / refs["travel"] + cfg.unassigned_penalty * unassigned)
    avail = np.array([max(st.close_slot - st.open_slot, 1) * st.ports for st in scn.stations])
    used = np.array([occupancy[st.station_id].clip(max=st.ports).sum() for st in scn.stations])
    return {
        "feasible": not violations and unassigned == 0,
        "valid_partial": not violations,       # no violation among served EVs
        "violations": violations,
        "energy_cost_inr": e_cost, "waiting_cost_inr": wait_cost, "travel_cost_inr": travel_cost,
        "total_cost_inr": e_cost + wait_cost + travel_cost,
        "objective": objective,
        "total_wait_h": w_h, "avg_wait_h": w_h / max(len(sessions), 1),
        "total_distance_km": dist_total,
        "utilization_pct": {sts[i].name: float(100 * used[k] / avail[k])
                            for k, i in enumerate(sorted(sts))},
        "avg_utilization_pct": float(100 * used.sum() / avail.sum()),
        "completion_pct": 100.0 * served / n if n else 100.0,
        "capacity_violations": cap_viol,
        "unassigned_count": unassigned,
        "runtime_s": runtime_s,
        "occupancy": occupancy.tolist(),
    }
