"""Automated tests for VoltQ. Run: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltq.classical import cpsat_schedule, full_residual, greedy_schedule, solve_cpsat
from voltq.data import generate_scenario
from voltq.evaluate import evaluate_sessions
from voltq.hybrid import HybridConfig, hybrid_schedule
from voltq.models import EV, Option, ScenarioConfig, Session, Station, assign_ports
from voltq.pipeline import pick_demonstrator
from voltq.preprocess import (build_options, charge_duration_slots, distance_km, energy_profile,
                              required_energy_kwh, travel_slots)
from voltq.qaoa import QAOAConfig, run_qaoa
from voltq.qubo import build_qubo, exact_solve, ising_energy, slack_weights, to_ising


@pytest.fixture(scope="module")
def world():
    scn = generate_scenario(ScenarioConfig())      # reproducible scenario: 10 EVs, seed 42
    return scn, build_options(scn)


def test_scenario_reproducible():
    a, b = generate_scenario(ScenarioConfig(seed=5)), generate_scenario(ScenarioConfig(seed=5))
    c = generate_scenario(ScenarioConfig(seed=6))
    assert a.evs == b.evs and a.stations == b.stations and a.prices == b.prices
    assert a.evs != c.evs


def test_energy_and_duration():
    ev = EV(0, "x", 60.0, 20.0, 80.0, 0, 10, 22.0, 0, 0)
    e = required_energy_kwh(ev, 0.9)
    assert e == pytest.approx(60 * 0.6 / 0.9)               # 40 kWh from grid
    assert charge_duration_slots(40.0, 22.0, 0.5) == 4      # 11 kWh/slot -> 3.64 -> 4
    assert charge_duration_slots(33.0, 22.0, 0.5) == 3      # exact fit, no extra slot
    prof = energy_profile(40.0, 22.0, 0.5, 4)
    assert prof == pytest.approx([11, 11, 11, 7]) and sum(prof) == pytest.approx(40.0)


def test_distance_and_travel():
    ev = EV(0, "x", 50, 10, 90, 0, 10, 11, 0.0, 0.0)
    st = Station(0, "S", 2, 22, 3.0, 4.0, 0, 12, 1.0)
    assert distance_km(ev, st) == pytest.approx(5.0)
    assert travel_slots(5.0, 40.0, 0.5) == 1                # 7.5 min -> 1 slot
    assert travel_slots(25.0, 40.0, 0.5) == 2               # 37.5 min -> 2 slots


def test_options_respect_windows(world):
    scn, pre = world
    evs = {e.ev_id: e for e in scn.evs}
    sts = {s.station_id: s for s in scn.stations}
    for o in pre.all_options():
        ev, st = evs[o.ev_id], sts[o.station_id]
        assert o.start >= ev.arrival_slot and o.start >= st.open_slot
        assert o.start + o.duration <= min(ev.departure_slot, st.close_slot)


def _toy():
    cfg = ScenarioConfig(n_evs=1, n_stations=1)
    scn = generate_scenario(cfg)
    return scn


def test_validator_flags_violations(world):
    scn, pre = world
    ev = scn.evs[0]
    o = pre.options[ev.ev_id][0]
    ok = [Session(ev.ev_id, o.station_id, 0, o.start, o.duration)]
    m = evaluate_sessions(scn, ok, pre.refs)
    assert m["valid_partial"] and m["capacity_violations"] == 0
    short = [Session(ev.ev_id, o.station_id, 0, o.start, o.duration - 1)]     # not enough energy
    assert not evaluate_sessions(scn, short, pre.refs)["valid_partial"]
    late = [Session(ev.ev_id, o.station_id, 0, ev.departure_slot, o.duration)]  # after departure
    assert not evaluate_sessions(scn, late, pre.refs)["valid_partial"]
    early = [Session(ev.ev_id, o.station_id, 0, ev.arrival_slot - 1, o.duration)]
    assert not evaluate_sessions(scn, early, pre.refs)["valid_partial"]
    assert evaluate_sessions(scn, ok, pre.refs)["unassigned_count"] == len(scn.evs) - 1


def test_capacity_and_overlap_detected(world):
    scn, pre = world
    st = scn.stations[0]
    # pick st.ports + 1 EVs with an option on station 0, force identical start -> over capacity
    picks = []
    for ev in scn.evs:
        o = next((o for o in pre.options[ev.ev_id] if o.station_id == st.station_id), None)
        if o:
            picks.append(o)
        if len(picks) == st.ports + 1:
            break
    t0 = max(o.start for o in picks)
    forced = {o.ev_id: Option(o.ev_id, o.station_id, t0, o.duration, o.energy_kwh, o.energy_cost,
                              o.wait_h, o.wait_cost, o.distance_km, o.travel_cost, o.cost) for o in picks}
    sessions = assign_ports(forced, scn.stations)
    assert any(s.port == -1 for s in sessions)
    m = evaluate_sessions(scn, sessions, pre.refs)
    assert m["capacity_violations"] > 0 and not m["feasible"]
    # explicit same-port overlap
    s1, s2 = sessions[0], sessions[1]
    clash = [Session(s1.ev_id, s1.station_id, 0, t0, s1.duration), Session(s2.ev_id, s2.station_id, 0, t0, s2.duration)]
    assert evaluate_sessions(scn, clash, pre.refs)["capacity_violations"] > 0


def test_port_assignment_is_conflict_free(world):
    scn, pre = world
    sch = greedy_schedule(scn, pre)
    sessions = sch.sessions(scn.stations)
    assert all(s.port >= 0 for s in sessions)
    by = {}
    for s in sessions:
        by.setdefault((s.station_id, s.port), []).append((s.start, s.start + s.duration))
    for ivs in by.values():
        ivs.sort()
        assert all(b[0] >= a[1] for a, b in zip(ivs, ivs[1:]))


@pytest.mark.parametrize("make", [greedy_schedule, cpsat_schedule])
def test_classical_schedules_feasible_and_objective_matches(world, make):
    scn, pre = world
    sch = make(scn, pre)
    m = evaluate_sessions(scn, sch.sessions(scn.stations), pre.refs)
    assert m["feasible"] and m["unassigned_count"] == 0 and m["completion_pct"] == 100.0
    option_sum = sum(o.cost for o in sch.assignments.values()) + scn.config.unassigned_penalty * len(sch.unassigned)
    assert m["objective"] == pytest.approx(option_sum, rel=1e-9)     # independent evaluator agrees


def test_cpsat_not_worse_than_greedy(world):
    scn, pre = world
    g = evaluate_sessions(scn, greedy_schedule(scn, pre).sessions(scn.stations), pre.refs)
    c = evaluate_sessions(scn, cpsat_schedule(scn, pre).sessions(scn.stations), pre.refs)
    assert c["objective"] <= g["objective"] + 1e-9


def test_slack_weights():
    for r in range(1, 9):
        w = slack_weights(r)
        sums = {sum(b * x for b, x in zip(w, bits)) for bits in np.ndindex(*(2,) * len(w))}
        assert sums == set(range(r + 1))


@pytest.fixture(scope="module")
def small_qubo(world):
    scn, pre = world
    sub = pick_demonstrator(scn, pre)
    return scn, pre, sub, build_qubo(sub, full_residual(scn))


def test_qubo_matches_ising_everywhere(small_qubo):
    q = small_qubo[3]
    h, J, c = to_ising(q)
    rng = np.random.default_rng(0)
    for _ in range(200):
        b = rng.integers(0, 2, q.n)
        assert q.energy(b) == pytest.approx(ising_energy(h, J, c, b), abs=1e-9)


def test_qubo_objective_correctness(small_qubo):
    """On feasible decision vectors with matching slack the QUBO energy equals the true cost."""
    scn, pre, sub, q = small_qubo
    ex = exact_solve(q)
    bits = ex["min_feasible_bits"]
    assert q.is_feasible(bits)
    assert ex["min_energy"] == pytest.approx(ex["min_feasible_energy"])      # penalties never beat feasibility
    assert ex["min_feasible_energy"] == pytest.approx(q.decision_cost(bits))
    # exact QUBO optimum equals CP-SAT optimum on the same options
    pen = scn.config.unassigned_penalty
    cp, status, obj = solve_cpsat(sub, full_residual(scn), pen)
    assert status == "OPTIMAL" and obj == pytest.approx(ex["min_feasible_energy"], abs=1e-3)


def test_qubo_penalises_violations(small_qubo):
    q = small_qubo[3]
    none = np.zeros(q.n, dtype=int)                                  # nobody assigned
    assert not q.is_feasible(none)
    assert q.energy(none) > exact_solve(q)["min_energy"]
    both = none.copy()
    g = next(iter(q.ev_groups.values()))
    both[g[0]] = both[g[1]] = 1                                      # one EV twice
    assert q.violations(both)["assignment"] >= 1 and q.energy(both) > exact_solve(q)["min_energy"]


def test_decode(small_qubo):
    q = small_qubo[3]
    bits = exact_solve(q)["min_feasible_bits"]
    chosen = q.decode(bits)
    assert sorted(o.ev_id for o in chosen) == sorted(q.ev_groups)
    assert sum(o.cost for o in chosen) == pytest.approx(q.decision_cost(bits))


def test_qaoa_runs_on_aer_and_is_reproducible(small_qubo):
    q = small_qubo[3]
    cfg = QAOAConfig(depth=1, shots=256, maxiter=15, seed=3)
    a, b = run_qaoa(q, cfg), run_qaoa(q, cfg)
    assert a.n_qubits == q.n and a.circuit_depth > 0 and a.n_evaluations >= 10
    assert a.best_energy >= exact_solve(q)["min_energy"] - 1e-9          # never below true optimum
    assert a.best_energy == pytest.approx(b.best_energy)
    assert a.history[0]["iter"] == 1 and len(a.history) == a.n_evaluations
    if a.best_feasible_bits is not None:
        assert q.is_feasible(a.best_feasible_bits)


def test_hybrid_schedule_valid(world):
    scn, pre = world
    raw, final, logs = hybrid_schedule(scn, pre, QAOAConfig(maxiter=20, shots=256), HybridConfig())
    for s in (raw, final):
        m = evaluate_sessions(scn, s.sessions(scn.stations), pre.refs)
        assert m["valid_partial"], m["violations"]
        assert m["capacity_violations"] == 0
    assert any(l.qaoa is not None for l in logs)
    mf = evaluate_sessions(scn, final.sessions(scn.stations), pre.refs)
    mr = evaluate_sessions(scn, raw.sessions(scn.stations), pre.refs)
    assert mf["objective"] <= mr["objective"] + 1e-9                       # refinement never hurts
