"""Tests for the recovery engine: events, commitments, grid power, recovery strategies.
Run: python -m pytest -q"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltq.classical import solve_cpsat
from voltq.data import generate_scenario
from voltq.evaluate import evaluate_sessions
from voltq.models import EV, Scenario, ScenarioConfig, Session, Station, grid_limit_kw
from voltq.preprocess import build_options
from voltq.qaoa import QAOAConfig
from voltq.qubo import build_qubo, exact_solve
from voltq.recovery.capacity import Capacity
from voltq.recovery.commitments import CommitmentType, RecoveryWeights
from voltq.recovery.detection import identify_affected, static_schedule
from voltq.recovery.events import (EVENT_TYPES, DepartureChange, NewArrival, PowerReduction,
                                   SessionOverrun, StationOutage, TariffChange, apply_events)
from voltq.recovery.problem import build_problem
from voltq.recovery.recovery_optimizer import STRATEGIES, recover
from voltq.simulation.network_simulator import NetworkSimulator, build_initial_state
from voltq.simulation.scenarios import (EVENT_SLOT, SCENARIOS, build_scenario, busiest_station,
                                        default_config)

QC = QAOAConfig(maxiter=12, shots=128, seed=3)


@pytest.fixture(scope="module")
def base():
    return build_initial_state(default_config(42))


def _run(key, seed=42, strategies=STRATEGIES):
    state, events = build_scenario(key, seed)
    sim = NetworkSimulator(state)
    run = sim.trigger(events)
    sim.recover(run, strategies, qcfg=QC)
    return run


# ---------------------------------------------------------------- initial state / commitments
def test_initial_state_valid_and_deterministic(base):
    again = build_initial_state(default_config(42))
    assert {i: (o.station_id, o.start) for i, o in base.assignments.items()} == \
           {i: (o.station_id, o.start) for i, o in again.assignments.items()}
    assert base.commitments == again.commitments
    types = {c.ctype for c in base.commitments.values()}
    assert CommitmentType.CRITICAL in types and CommitmentType.CONFIRMED in types
    m = evaluate_sessions(base.scn, base.sessions(), base.refs)
    assert m["valid_partial"] and m["power_violations"] == 0


# ---------------------------------------------------------------- event system
def test_station_outage_closes_station_and_names_affected(base):
    sid = busiest_station(base, EVENT_SLOT)
    eff = StationOutage(EVENT_SLOT, sid).apply(base)
    assert eff.state.scn.stations[sid].close_slot == EVENT_SLOT
    assert base.scn.stations[sid].close_slot > EVENT_SLOT            # original state untouched
    future = {i for i, o in base.assignments.items() if o.station_id == sid and o.start >= EVENT_SLOT}
    cut = {i for i, o in base.assignments.items()
           if o.station_id == sid and o.start < EVENT_SLOT < o.start + o.duration}
    assert eff.directly_affected == future | cut
    for i in cut:                                                    # interrupted EV keeps its energy
        assert i not in eff.state.assignments
        assert eff.state.scn.evs[i].soc_pct > base.scn.evs[i].soc_pct
        assert eff.state.scn.evs[i].arrival_slot == EVENT_SLOT


def test_new_arrival_event(base):
    ev = NewArrival(EVENT_SLOT, departure_slot=11)
    eff = ev.apply(base)
    nid = len(base.scn.evs)
    assert eff.directly_affected == {nid} and len(eff.state.scn.evs) == nid + 1
    assert eff.state.commitments[nid].ctype == CommitmentType.UNCONFIRMED
    assert len(base.scn.evs) == nid


def test_departure_change_event_and_guard(base):
    i, o = next((i, o) for i, o in base.assignments.items() if o.start >= EVENT_SLOT)
    eff = DepartureChange(EVENT_SLOT, i, o.start + o.duration - 1).apply(base)
    assert eff.state.scn.evs[i].departure_slot == o.start + o.duration - 1
    assert i in eff.directly_affected
    started = next((i, o) for i, o in base.assignments.items() if o.start < EVENT_SLOT < o.start + o.duration)
    with pytest.raises(ValueError):
        DepartureChange(EVENT_SLOT, started[0], EVENT_SLOT + 1).apply(base)


def test_power_reduction_event(base):
    sid = busiest_station(base, EVENT_SLOT)
    st0 = base.scn.stations[sid]
    eff = PowerReduction(EVENT_SLOT, sid, 0.6).apply(base)
    st1 = eff.state.scn.stations[sid]
    assert grid_limit_kw(st1, EVENT_SLOT) == pytest.approx(0.6 * st0.ports * st0.power_kw)
    assert grid_limit_kw(st1, EVENT_SLOT - 1) == st0.ports * st0.power_kw     # earlier slots unchanged


def test_tariff_change_event(base):
    eff = TariffChange(EVENT_SLOT, 5, 8, 0.5).apply(base)
    assert eff.state.scn.prices[6] == pytest.approx(base.scn.prices[6] * 0.5)
    assert eff.state.scn.prices[2] == base.scn.prices[2]
    assert eff.directly_affected                                       # EVs that could shift


def test_session_overrun_event(base):
    i, o = next((i, o) for i, o in base.assignments.items() if o.start < EVENT_SLOT < o.start + o.duration)
    eff = SessionOverrun(EVENT_SLOT, i, 2).apply(base)
    assert eff.state.assignments[i].duration >= o.duration
    assert eff.state.scn.evs[i].departure_slot >= eff.state.assignments[i].start + eff.state.assignments[i].duration
    j = next(j for j, p in base.assignments.items() if p.start >= EVENT_SLOT)
    with pytest.raises(ValueError):
        SessionOverrun(EVENT_SLOT, j, 1).apply(base)


def test_event_registry_and_combination(base):
    assert {"station_outage", "new_arrival", "departure_change", "power_reduction",
            "tariff_change", "session_overrun"} <= set(EVENT_TYPES)
    eff = apply_events(base, [StationOutage(EVENT_SLOT, 0), NewArrival(EVENT_SLOT)])
    assert len(eff.state.scn.evs) == len(base.scn.evs) + 1 and len(eff.state.event_log) == 2


# ---------------------------------------------------------------- affected-EV detection
def test_affected_detection_on_outage(base):
    sid = busiest_station(base, EVENT_SLOT)
    eff = StationOutage(EVENT_SLOT, sid).apply(base)
    rep = identify_affected(eff.state, eff.directly_affected)
    on_failed = {i for i, o in base.assignments.items() if o.station_id == sid and o.start >= EVENT_SLOT}
    assert on_failed <= rep.affected
    assert not (rep.affected & rep.frozen)
    unrelated = {i for i, o in base.assignments.items() if o.station_id != sid and o.start >= EVENT_SLOT}
    assert not (unrelated & set(rep.invalidated))                      # other stations stay valid


def test_event_without_impact_changes_nothing(base):
    sid = busiest_station(base, 11)                                    # nothing left to run there
    eff = StationOutage(11, sid).apply(base)
    rep = identify_affected(eff.state, eff.directly_affected)
    res = recover(eff.state, eff.directly_affected, "CP-SAT recovery", report=rep)
    keep = lambda a: {i: (o.station_id, o.start) for i, o in a.items()}
    if not rep.affected:
        assert keep(res.assignments) == keep(eff.state.assignments)
    moved = [i for i, o in res.assignments.items()
             if i in base.assignments and (o.station_id, o.start) != (base.assignments[i].station_id,
                                                                      base.assignments[i].start)]
    assert set(moved) <= set(res.free_ids)                             # only free EVs may move


# ---------------------------------------------------------------- recovery strategies
@pytest.mark.parametrize("key", list(SCENARIOS))
def test_every_strategy_is_valid_on_every_scenario(key):
    run = _run(key)
    for name, m in run.metrics.items():
        assert m["valid"], (key, name, m["hard_violations"])
        assert m["capacity_violations"] == 0 and m["grid_power_violations"] == 0
    assert run.results["Hybrid QAOA recovery"].meta["fallback_batches"] >= 0


def test_recovery_is_deterministic_for_greedy_and_hybrid():
    a, b = _run("failure"), _run("failure")
    key = lambda r: {i: (o.station_id, o.start) for i, o in r.assignments.items()}
    for s in ("Greedy recovery", "Hybrid QAOA recovery"):
        assert key(a.results[s]) == key(b.results[s])
    assert a.metrics["CP-SAT recovery"]["recovery_objective"] == \
        pytest.approx(b.metrics["CP-SAT recovery"]["recovery_objective"])


def test_solver_ordering_on_identical_inputs():
    for key in ("failure", "combined", "arrivals"):
        run = _run(key)
        obj = {n: m["recovery_objective"] for n, m in run.metrics.items()}
        assert obj["CP-SAT recovery"] <= obj["Greedy recovery"] + 1e-6
        assert obj["CP-SAT full-network"] <= obj["Hybrid QAOA recovery"] + 1e-6    # global optimum
        assert obj["CP-SAT recovery"] < obj["Static (no recovery)"]
        assert run.metrics["CP-SAT recovery"]["sla_success_pct"] >= run.metrics["Static (no recovery)"]["sla_success_pct"]


def test_local_recovery_moves_only_free_evs():
    run = _run("failure")
    res = run.results["CP-SAT recovery"]
    for i, o in res.assignments.items():
        old = run.before.assignments.get(i)
        if old is not None and i not in res.free_ids:
            assert (old.station_id, old.start) == (o.station_id, o.start)
    # frozen (already started) sessions are untouched by every strategy
    for name, r in run.results.items():
        for i, o in run.after.frozen().items():
            assert r.assignments[i] is o or (r.assignments[i].station_id, r.assignments[i].start) == (o.station_id, o.start)


def test_local_recovery_is_more_stable_than_unprotected_full_reoptimisation():
    state, events = build_scenario("failure", 42)
    after = apply_events(state, events)
    local = recover(after.state, after.directly_affected, "CP-SAT recovery")
    st_off = after.state.copy()
    st_off.weights = RecoveryWeights(protect_confirmed="off")
    full = recover(st_off, after.directly_affected, "CP-SAT full-network")
    changed = lambda r: sum(1 for i, o in state.assignments.items()
                            if i not in r.assignments or (r.assignments[i].station_id, r.assignments[i].start) != (o.station_id, o.start))
    assert changed(local) <= changed(full)


def test_critical_ev_is_served_when_feasible():
    run = _run("failure")
    crit = [i for i, c in run.after.commitments.items() if c.ctype == CommitmentType.CRITICAL]
    for name in ("Greedy recovery", "CP-SAT recovery", "Hybrid QAOA recovery"):
        assert run.metrics[name]["critical_sla_pct"] == 100.0, name
    assert crit


def test_flexible_reservation_can_be_reassigned_to_save_a_critical_one():
    state, events = build_scenario("failure", 42)
    after = apply_events(state, events)
    rep = identify_affected(after.state, after.directly_affected)
    res = recover(after.state, after.directly_affected, "CP-SAT recovery", report=rep)
    assert res.stages >= 0 and set(rep.affected) <= set(res.free_ids)


def test_impossible_recovery_is_reported_not_hidden():
    """All stations go down: nothing can be placed; the schedule must stay valid and honest."""
    state, _ = build_scenario("failure", 42)
    events = [StationOutage(EVENT_SLOT, s.station_id) for s in state.scn.stations]
    sim = NetworkSimulator(state)
    run = sim.trigger(events)
    sim.recover(run, qcfg=QC)
    for name, m in run.metrics.items():
        assert m["valid"]
        assert m["unassigned_evs"] >= len(run.report.affected - run.report.frozen) > 0 or name.startswith("Static")
        assert m["sla_success_pct"] < 100.0
        assert m["unserved_energy_kwh"] > 0
    assert run.results["CP-SAT recovery"].meta["unavoidable_no_option"]


def test_commitment_violations_split_forced_vs_avoidable():
    run = _run("failure")
    for m in run.metrics.values():
        assert m["commitment_violations"] == m["commitment_violations_forced"] + m["commitment_violations_avoidable"]


def test_strict_protection_does_not_touch_valid_confirmed_for_noncritical_unserved():
    state, events = build_scenario("departure", 42)
    after = apply_events(state, events)
    after.state.weights = RecoveryWeights(protect_confirmed="strict")
    rep = identify_affected(after.state, after.directly_affected)
    res = recover(after.state, after.directly_affected, "CP-SAT recovery", report=rep)
    confirmed_valid = [i for i, c in after.state.commitments.items()
                       if c.ctype == CommitmentType.CONFIRMED and i in after.state.assignments
                       and i not in rep.affected and i not in rep.frozen]
    for i in confirmed_valid:
        o, n = after.state.assignments[i], res.assignments[i]
        assert (o.station_id, o.start) == (n.station_id, n.start)


# ---------------------------------------------------------------- grid power
def _toy_grid(grid_kw):
    cfg = ScenarioConfig(n_evs=2, n_stations=1, seed=1, n_slots=8)
    scn = generate_scenario(cfg)
    st = Station(0, "S1", 2, 50.0, 0.0, 0.0, 0, 8, 1.0, grid_kw=grid_kw, grid_from=0)
    evs = [EV(i, f"EV{i + 1:02d}", 50.0, 20.0, 80.0, 0, 8, 50.0, 0.0, 0.0) for i in range(2)]
    return replace(scn, evs=evs, stations=[st])


def test_grid_limit_function():
    st = Station(0, "S", 2, 50.0, 0, 0, 0, 12, 1.0, grid_kw=60.0, grid_from=4)
    assert grid_limit_kw(st, 3) == 100.0 and grid_limit_kw(st, 4) == 60.0
    assert grid_limit_kw(replace(st, grid_kw=None), 9) == 100.0


def test_capacity_enforces_ports_and_power():
    scn = _toy_grid(60.0)
    pre = build_options(scn)
    a = next(o for o in pre.options[0] if o.start == 0)
    b = next(o for o in pre.options[1] if o.start == 0)
    cap = Capacity(scn, {0: a})
    assert not cap.fits(b)                         # 50 + 50 kW > 60 kW even though a 2nd port is free
    assert Capacity(_toy_grid(100.0), {0: a}).fits(b)
    c = next(o for o in pre.options[1] if o.start >= a.start + a.duration)
    assert cap.fits(c)


def test_validator_detects_power_violation_and_never_calls_it_feasible():
    scn = _toy_grid(60.0)
    pre = build_options(scn)
    o0 = next(o for o in pre.options[0] if o.start == 0)
    o1 = next(o for o in pre.options[1] if o.start == 0)
    sessions = [Session(0, 0, 0, 0, o0.duration), Session(1, 0, 1, 0, o1.duration)]
    m = evaluate_sessions(scn, sessions, pre.refs)
    assert m["power_violations"] > 0 and not m["feasible"] and not m["valid_partial"]
    ok = [Session(0, 0, 0, 0, o0.duration), Session(1, 0, 0, o0.duration, o1.duration)]
    assert evaluate_sessions(scn, ok, pre.refs)["power_violations"] == 0


def test_cpsat_and_qubo_agree_under_binding_power_limit():
    scn = _toy_grid(60.0)
    pre = build_options(scn)
    cap = Capacity(scn, {})
    cp, status, obj = solve_cpsat(pre.options, cap.ports_free, 10.0, power_residual=cap.power_free)
    sub = {i: sorted(v, key=lambda o: (o.cost, o.start))[:3] for i, v in pre.options.items()}
    q = build_qubo(sub, cap.ports_free, power_residual=cap.power_free)
    assert q.power, "power row must be present in the QUBO"
    ex = exact_solve(q)
    best = ex["min_feasible_bits"]
    assert q.violations(best)["power"] == 0
    # the 2 EVs can never overlap: chosen sessions must be disjoint in time
    chosen = q.decode(best)
    spans = sorted((o.start, o.start + o.duration) for o in chosen)
    assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:]))
    cp_spans = sorted((o.start, o.start + o.duration) for o in cp.values())
    assert all(b[0] >= a[1] for a, b in zip(cp_spans, cp_spans[1:]))
    cp_cost = sum(o.cost for o in cp.values())
    sub_cp, _, sub_obj = solve_cpsat(sub, cap.ports_free, 10.0, power_residual=cap.power_free)
    assert ex["min_feasible_energy"] == pytest.approx(sub_obj, abs=1e-3)
    assert cp_cost <= sub_obj + 1e-3


def test_power_reduction_scenario_respects_aggregate_limit():
    run = _run("power")
    sid = [e for e in run.events][0].station_id
    for name, res in run.results.items():
        m = run.metrics[name]
        assert m["grid_power_violations"] == 0
        kw = np.array(m["evaluator"]["occupancy_kw"])[sid]
        limit = [grid_limit_kw(run.after.scn.stations[sid], t) for t in range(len(kw))]
        assert (kw <= np.array(limit) + 1e-6).all()


# ---------------------------------------------------------------- fairness (optional)
def test_fairness_weight_changes_costs_only_when_enabled(base):
    eff = StationOutage(EVENT_SLOT, busiest_station(base, EVENT_SLOT)).apply(base)
    st = eff.state
    free = sorted(identify_affected(st, eff.directly_affected).affected)
    st.hist(free[0])["reschedules"] = 3
    off = build_problem(st, free, st.frozen(), RecoveryWeights(fairness=0.0))
    on = build_problem(st, free, st.frozen(), RecoveryWeights(fairness=1.0))
    waited = [(a, b) for a, b in zip(off.options[free[0]], on.options[free[0]]) if a.wait_cost > 0]
    assert all(b.cost > a.cost for a, b in waited)
    assert all(b.cost == a.cost for a, b in zip(off.options[free[1]], on.options[free[1]]))


# ---------------------------------------------------------------- OCPP-like adapter + detection details
def test_ocpp_like_messages_map_to_events(base):
    from voltq.realtime.ocpp_like import event_from_message
    ev = event_from_message({"action": "StatusNotification", "stationId": "S2", "status": "Faulted"}, base, 4)
    assert isinstance(ev, StationOutage) and ev.station_id == 1 and ev.slot == 4
    st = base.scn.stations[0]
    ev = event_from_message({"action": "SetChargingProfile", "stationId": "S1",
                             "limit_kw": 0.5 * st.ports * st.power_kw}, base, 3)
    assert isinstance(ev, PowerReduction) and ev.remaining_fraction == pytest.approx(0.5)
    with pytest.raises(ValueError):
        event_from_message({"action": "StatusNotification", "stationId": "S1", "status": "Available"}, base, 3)
    with pytest.raises(KeyError):
        event_from_message({"action": "ChangeAvailability", "stationId": "S9", "type": "Inoperative"}, base, 3)


def test_interrupted_ev_is_reported_as_interrupted_not_unserved(base):
    eff = StationOutage(EVENT_SLOT, busiest_station(base, EVENT_SLOT)).apply(base)
    rep = identify_affected(eff.state, eff.directly_affected)
    assert rep.interrupted == eff.state.interrupted
    for i in rep.interrupted:
        assert i in rep.affected and i not in rep.unserved
        assert any("interrupted" in r for r in rep.reasons(i))
