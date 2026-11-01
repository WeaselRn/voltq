"""Recovery metrics, all computed from the FINAL schedule (never copied from solver objectives).

Hard-rule checks come from the independent validator (`evaluate_sessions`). Soft effects
(reassignment, commitments) are counted against the schedule that existed before the event.
"""
from __future__ import annotations

import numpy as np

from ..evaluate import evaluate_sessions
from ..models import Option, assign_ports
from ..preprocess import required_energy_kwh
from .commitments import CommitmentType
from .detection import AffectedReport
from .problem import network_objective
from .recovery_optimizer import RecoveryResult
from .state import NetworkState

PROTECTED = (CommitmentType.CONFIRMED, CommitmentType.CRITICAL)


def jain_index(waits_h: list[float]) -> float:
    """Jain fairness of per-EV service quality 1/(1+wait_h): 1.0 = everyone is treated alike,
    1/n = one EV bears all the waiting."""
    a = 1.0 / (1.0 + np.array(waits_h, dtype=float))
    if a.size == 0:
        return 1.0
    return float(a.sum() ** 2 / (a.size * (a ** 2).sum()))


def compute_metrics(before: NetworkState, after: NetworkState, result: RecoveryResult,
                    report: AffectedReport) -> dict:
    scn = after.scn
    sessions = assign_ports(result.assignments, scn.stations)
    m = evaluate_sessions(scn, sessions, after.refs, result.runtime_s)
    evs = {e.ev_id: e for e in scn.evs}
    served = {s.ev_id for s in sessions}
    forced = set(report.invalidated) | set(after.interrupted)

    moved, dropped, viol_forced, viol_avoid = [], [], [], []
    for i, old in before.assignments.items():
        new = result.assignments.get(i)
        changed = new is None or (new.station_id, new.start) != (old.station_id, old.start)
        if not changed:
            continue
        (dropped if new is None else moved).append(i)
        if before.commitments[i].ctype in PROTECTED:
            (viol_forced if i in forced else viol_avoid).append(i)
    waits = []
    for s in sessions:
        o = result.assignments[s.ev_id]
        waits.append(o.wait_h)
    unserved = [e.ev_id for e in scn.evs if e.ev_id not in served]
    unserved_kwh = sum(required_energy_kwh(evs[i], scn.config.efficiency) for i in unserved)
    crit = [e.ev_id for e in scn.evs if after.commitments[e.ev_id].ctype == CommitmentType.CRITICAL]
    ok_set = served if m["valid_partial"] else set()
    obj = network_objective(after, result.assignments)
    n = len(scn.evs)
    return {
        "strategy": result.strategy,
        "recovery_runtime_s": result.runtime_s,
        "total_cost_inr": m["total_cost_inr"],
        "energy_cost_inr": m["energy_cost_inr"],
        "waiting_cost_inr": m["waiting_cost_inr"],
        "travel_cost_inr": m["travel_cost_inr"],
        "cost_per_served_inr": m["total_cost_inr"] / max(len(sessions), 1),
        "avg_wait_min": 60 * m["total_wait_h"] / max(len(sessions), 1),
        "max_wait_min": 60 * (max((s_ for s_ in waits), default=0.0)),
        "reassigned_reservations": len(moved) + len(dropped),
        "moved": len(moved), "dropped": len(dropped),
        "commitment_violations": len(viol_forced) + len(viol_avoid),
        "commitment_violations_forced": len(viol_forced),
        "commitment_violations_avoidable": len(viol_avoid),
        "sla_success_pct": 100.0 * len(ok_set) / n if n else 100.0,
        "critical_sla_pct": 100.0 * len([i for i in crit if i in ok_set]) / len(crit) if crit else 100.0,
        "unassigned_evs": len(unserved),
        "unserved_evs": [evs[i].name for i in unserved],
        "unserved_energy_kwh": unserved_kwh,
        "avg_utilization_pct": m["avg_utilization_pct"],
        "grid_power_violations": m["power_violations"],
        "capacity_violations": m["capacity_violations"],
        "hard_violations": m["violations"],
        "valid": m["valid_partial"],
        "fairness_jain": jain_index(waits),
        "wait_std_min": 60 * float(np.std(waits)) if waits else 0.0,
        "recovery_objective": obj["total"],
        "objective_terms": obj,
        "affected_evs": len(result.free_ids) if result.free_ids else len(report.affected),
        "expansion_stages": result.stages,
        "evaluator": m,
    }


def metrics_frame(rows: list[dict]):
    import pandas as pd
    keep = ["strategy", "recovery_runtime_s", "total_cost_inr", "avg_wait_min", "max_wait_min",
            "reassigned_reservations", "commitment_violations", "sla_success_pct", "critical_sla_pct",
            "unassigned_evs", "unserved_energy_kwh", "avg_utilization_pct", "grid_power_violations",
            "fairness_jain", "recovery_objective", "valid"]
    return pd.DataFrame([{k: r[k] for k in keep} for r in rows]).set_index("strategy")
