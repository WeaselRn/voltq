"""VoltQ - adaptive EV charging coordination platform.  Run: streamlit run app.py

Every number on screen comes from the optimisation engine (voltq/). Quantum circuits are simulated
locally on Qiskit Aer; no IBM hardware or cloud is used, no live charger is contacted.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from voltq import viz
from voltq.hybrid import HybridConfig
from voltq.models import ScenarioConfig, assign_ports, grid_limit_kw
from voltq.pipeline import qaoa_summary, schedule_table
from voltq.qaoa import OPTIMIZERS, QAOAConfig, build_circuit
from voltq.qubo import to_ising
from voltq.recovery import viz as rviz
from voltq.recovery.commitments import CommitmentType, RecoveryWeights, assign_commitments
from voltq.recovery.events import (DepartureChange, NewArrival, PowerReduction, SessionOverrun,
                                   StationOutage, TariffChange, in_progress)
from voltq.recovery.recovery_metrics import metrics_frame
from voltq.recovery.recovery_optimizer import STRATEGIES
from voltq.recovery.state import NetworkState
from voltq.simulation.benchmark import run_benchmark, summarize
from voltq.simulation.network_simulator import NetworkSimulator
from voltq.simulation.scenarios import EVENT_SLOT, SCENARIOS, arrival_near
from voltq.study import best_method, run_all

st.set_page_config(page_title="VoltQ", page_icon="⚡", layout="wide")
plt.rcParams.update({"figure.facecolor": rviz.BG, "axes.facecolor": rviz.PANEL, "axes.edgecolor": rviz.GRID,
                     "axes.labelcolor": rviz.MUTED, "text.color": rviz.TEXT, "xtick.color": rviz.MUTED,
                     "ytick.color": rviz.MUTED, "axes.titlecolor": rviz.TEXT, "grid.color": rviz.GRID,
                     "legend.labelcolor": rviz.MUTED, "savefig.facecolor": rviz.BG})
st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1500px}
[data-testid=stMetric]{background:#16202c;border:1px solid #2a3646;border-radius:10px;padding:10px 14px}
[data-testid=stMetricLabel]{color:#8b98a9}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.8rem;font-weight:600}
.ok{background:#14532d;color:#86efac}.warn{background:#713f12;color:#fde68a}.bad{background:#7f1d1d;color:#fecaca}
.cls{background:#1e293b;color:#cbd5e1}
.event{border-left:4px solid #e879f9;background:#16202c;padding:8px 14px;border-radius:6px;margin-bottom:6px}
</style>""", unsafe_allow_html=True)

st.title("⚡ VoltQ")
st.caption("Adaptive EV charging coordination: recover the charging network, not just the mathematical "
           "solution. Quantum circuits (QAOA) are simulated on this computer with Qiskit Aer; "
           "no quantum hardware, cloud or live charger is used.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Network")
    n_evs = st.slider("Number of EVs", 4, 300, 12)
    n_st = st.slider("Number of stations", 2, 75, 4)
    st.caption("Large sizes are slow: about 1 station per 4 EVs keeps it realistic.")
    pmin, pmax = st.slider("Ports per station (min, max)", 1, 4, (2, 2))
    n_slots = st.slider("Time slots (30 min each)", 8, 48, 12)
    seed = st.number_input("Random seed", 0, 10_000, 42)
    initial = st.selectbox("Initial schedule from", ["OR-Tools", "Greedy", "Hybrid (QAOA + refinement)"])
    with st.expander("Electricity price (₹ per kWh)"):
        peak = st.number_input("Peak", 1.0, 50.0, 14.0, 0.5)
        normal = st.number_input("Normal", 1.0, 50.0, 9.0, 0.5)
        off = st.number_input("Off-peak", 1.0, 50.0, 5.0, 0.5)
        start_hour = st.slider("Window starts at (hour of day)", 0.0, 23.5, 19.0, 0.5)
    with st.expander("What matters more? (weights)"):
        alpha = st.slider("Electricity cost", 0.0, 3.0, 1.0, 0.1)
        beta = st.slider("Waiting time", 0.0, 3.0, 1.0, 0.1)
        gamma = st.slider("Travel distance", 0.0, 3.0, 1.0, 0.1)
    with st.expander("Recovery policy"):
        protect = st.selectbox("Confirmed-reservation protection", ["soft", "strict", "off"],
                               help="soft: confirmed EVs move only as a last resort, with a penalty. "
                                    "strict: only moved to save a critical EV. off: no protection.")
        lam_re = st.slider("Penalty for moving any reservation", 0.0, 3.0, 0.6, 0.1)
        lam_conf = st.slider("Extra penalty: confirmed reservation", 0.0, 6.0, 2.0, 0.5)
        lam_fair = st.slider("Fairness weight (0 = off)", 0.0, 3.0, 0.0, 0.25,
                             help="Raises the cost of waiting for EVs that were already moved or delayed.")
    with st.expander("Quantum (QAOA) settings"):
        depth = st.slider("Circuit layers (p)", 1, 3, 1)
        shots = st.select_slider("Samples per run (shots)", [128, 256, 512, 1024], 256)
        optimizer = st.selectbox("Tuning method", OPTIMIZERS)
        maxiter = st.slider("Tuning steps", 10, 120, 40, 10)
        cvar = st.slider("Focus on best samples (1 = off)", 0.1, 1.0, 1.0, 0.1)
        batch = st.slider("EVs per quantum batch", 2, 4, 3)
        sweep = st.checkbox("Also compare 1, 2 and 3 layers (slower)", False)
    run = st.button("Build network / run optimization", type="primary", width="stretch")


@st.cache_resource(show_spinner="Optimizing (simulating quantum circuits)…")
def cached_run(cfg_t, q_t, b, sweep_flag):
    cfg = ScenarioConfig(**dict(cfg_t))
    qcfg = QAOAConfig(**dict(q_t))
    return run_all(cfg, qcfg, HybridConfig(batch_size=b), do_sweep=sweep_flag)


qcfg = QAOAConfig(depth=depth, shots=shots, optimizer=optimizer, maxiter=maxiter, seed=int(seed),
                  cvar_alpha=cvar)
hcfg = HybridConfig(batch_size=batch)
weights = RecoveryWeights(lam_reassign=lam_re, fairness=lam_fair, protect_confirmed=protect)
weights.lam_commit[CommitmentType.CONFIRMED] = lam_conf

if run or "last" not in st.session_state:
    cfg_t = tuple(dict(n_evs=n_evs, n_stations=n_st, ports_min=pmin, ports_max=pmax, n_slots=n_slots,
                       seed=int(seed), peak_price=peak, normal_price=normal, offpeak_price=off,
                       start_hour=start_hour, alpha=alpha, beta=beta, gamma=gamma).items())
    q_t = tuple(dict(depth=depth, shots=shots, optimizer=optimizer, maxiter=maxiter,
                     seed=int(seed), cvar_alpha=cvar).items())
    r = cached_run(cfg_t, q_t, batch, sweep)
    sched = r["schedules"][initial]
    commit = assign_commitments(r["scenario"], r["pre"], sched.assignments, int(seed))
    st.session_state["last"] = r
    st.session_state["sim"] = NetworkSimulator(NetworkState(r["scenario"], r["pre"].refs,
                                                            dict(sched.assignments), commit, 0))
    st.session_state["base_state"] = st.session_state["sim"].state
    st.session_state.pop("run", None)
res = st.session_state["last"]
sim: NetworkSimulator = st.session_state["sim"]
sim.state.weights = weights
scn0, pre, evals, scheds, red = res["scenario"], res["pre"], res["evals"], res["schedules"], res["reduced"]


def clock(scn, slot: int) -> str:
    return rviz.slot_labels(scn)[min(slot, scn.config.n_slots)]


def reservation_table(state: NetworkState, assignments=None) -> pd.DataFrame:
    scn = state.scn
    assignments = state.assignments if assignments is None else assignments
    ports = {s.ev_id: s.port for s in assign_ports(assignments, scn.stations)}
    rows = []
    for ev in scn.evs:
        o = assignments.get(ev.ev_id)
        c = state.commitments[ev.ev_id]
        rows.append({
            "EV": ev.name, "Commitment": c.ctype.value, "Status": "scheduled" if o else "UNASSIGNED",
            "Station": scn.stations[o.station_id].name if o else "-",
            "Port": ports[ev.ev_id] + 1 if o else None,
            "Charge from": clock(scn, o.start) if o else "-",
            "Charge to": clock(scn, o.start + o.duration) if o else "-",
            "Energy (kWh)": round(o.energy_kwh, 1) if o else None,
            "Wait (min)": round(o.wait_h * 60) if o else None,
            "Departs": clock(scn, ev.departure_slot),
            "Slack (slots)": c.flexibility_slots})
    return pd.DataFrame(rows)


def down_map(before: NetworkState, after: NetworkState) -> dict[int, int]:
    return {s.station_id: s.close_slot for s in after.scn.stations
            if s.close_slot < before.scn.stations[s.station_id].close_slot}


def tag(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{text}</span>'


tabs = st.tabs(["Network overview", "Charging schedule", "Disruption simulator", "Recovery comparison",
                "Quantum engine", "Benchmarks"])

# ------------------------------------------------------------------ 1. network overview
with tabs[0]:
    state = sim.state
    scn = state.scn
    m = rviz and None
    from voltq.evaluate import evaluate_sessions
    ev_m = evaluate_sessions(scn, state.sessions(), state.refs)
    unassigned = [e for e in scn.evs if e.ev_id not in state.assignments]
    down = {s.name for s in scn.stations if s.close_slot < scn.config.n_slots
            and s.close_slot <= state.t_now} if state.event_log else set()
    capped = [s.name for s in scn.stations if s.grid_kw is not None]
    status = (tag("DEGRADED", "bad") if down or unassigned else tag("LIMITED", "warn") if capped
              else tag("NOMINAL", "ok"))
    st.markdown(f"**Network status** {status} &nbsp; "
                + (f"Out of service: {', '.join(sorted(down))}. " if down else "")
                + (f"Grid-capped: {', '.join(capped)}. " if capped else "")
                + (f"{len(unassigned)} EV(s) without a charging slot." if unassigned else ""),
                unsafe_allow_html=True)
    c = st.columns(6)
    c[0].metric("EVs", len(scn.evs))
    c[1].metric("Stations", len(scn.stations))
    c[2].metric("Active reservations", len(state.assignments))
    c[3].metric("Available ports", sum(s.ports for s in scn.stations if s.name not in down))
    c[4].metric("Grid power now", f"{sum(grid_limit_kw(s, state.t_now) for s in scn.stations if s.name not in down):.0f} kW")
    c[5].metric("Utilization", f"{ev_m['avg_utilization_pct']:.0f}%")
    c = st.columns(4)
    c[0].metric("Unassigned EVs", len(unassigned))
    c[1].metric("Total cost", f"₹{ev_m['total_cost_inr']:,.0f}")
    c[2].metric("Average wait", f"{ev_m['avg_wait_h'] * 60:.0f} min")
    c[3].metric("Constraint check", "PASS" if ev_m["valid_partial"] else "FAIL")
    if pre.infeasible_evs:
        st.warning("These EVs cannot be charged in time at any station: "
                   + ", ".join(scn0.evs[i].name for i in pre.infeasible_evs))
    l, r_ = st.columns(2)
    with l:
        st.pyplot(rviz.fig_network_utilization(scn, ev_m["occupancy"], "Port occupancy (% of ports busy)"))
    with r_:
        st.pyplot(viz.fig_tariff(scn))
    st.caption("Utilization uses the current schedule of the network (after any adopted recovery).")

# ------------------------------------------------------------------ 2. charging schedule
with tabs[1]:
    state = sim.state
    st.markdown("**In short:** every row is one EV reservation. Colour shows how strongly it is committed: "
                "red = critical departure, green = confirmed, amber = flexible, blue = new / unconfirmed.")
    bad = evaluate_sessions(state.scn, state.sessions(), state.refs)
    if bad["violations"]:
        st.error("Validator found problems:\n" + "\n".join(bad["violations"]))
    else:
        st.success("Independent validator: no hard rule is broken (ports, grid power, windows, energy).")
    st.pyplot(rviz.fig_gantt(state.scn, state.assignments, {}, state.commitments,
                             f"Charging schedule ({initial if not state.event_log else 'after recovery'})",
                             t_now=state.t_now if state.event_log else None))
    st.dataframe(reservation_table(state), hide_index=True, width="stretch")
    with st.expander("Plans from every scheduler (initial network)"):
        pick = st.selectbox("Plan to view", list(scheds), index=list(scheds).index(best_method(evals)))
        st.pyplot(viz.fig_schedule(scn0, scheds[pick].sessions(scn0.stations), f"Charging timeline: {pick}"))
        tbl = schedule_table(scn0, scheds[pick])[["ev", "status", "station", "port", "start_h", "required_kwh",
                                                  "energy_cost_inr", "wait_h", "distance_km"]]
        tbl.columns = ["EV", "Status", "Station", "Port", "Starts at (hour)", "Energy needed (kWh)",
                       "Electricity cost (₹)", "Wait (h)", "Drive (km)"]
        st.dataframe(tbl, width="stretch", hide_index=True)

# ------------------------------------------------------------------ 3. disruption simulator
with tabs[2]:
    state = sim.state
    scn = state.scn
    st.markdown("**Workflow:** pick a disruption → *Trigger* → inspect affected EVs → *Run recovery* → "
                "compare before and after. Only sessions that have not started yet can change; sessions "
                "already charging are frozen (or interrupted by the event itself).")
    mode = st.radio("Disruption", ["Predefined scenario", "Custom event"], horizontal=True)
    events = []
    err = None
    if mode == "Predefined scenario":
        key = st.selectbox("Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k].title)
        st.caption(SCENARIOS[key].blurb + f" Occurs at slot {EVENT_SLOT} ({clock(scn, EVENT_SLOT)}).")
        try:
            events = SCENARIOS[key].build(state, int(seed))
        except (ValueError, KeyError, StopIteration) as exc:
            err = f"This scenario cannot be built on the current network: {exc}"
    else:
        kinds = ["Station outage", "New EV arrival", "Departure time change", "Station power reduction",
                 "Tariff change", "Charging session overrun"]
        kind = st.selectbox("Event type", kinds)
        slot = st.slider("Event happens at slot", 1, scn.config.n_slots - 2, EVENT_SLOT)
        stn = {s.station_id: s.name for s in scn.stations}
        if kind == "Station outage":
            events = [StationOutage(slot, st.selectbox("Station", list(stn), format_func=stn.get))]
        elif kind == "New EV arrival":
            cc = st.columns(4)
            near = cc[0].selectbox("Arrives near", list(stn), format_func=stn.get)
            base = arrival_near(state, near, slot, 0, int(seed))
            kwh = cc[1].selectbox("Battery (kWh)", [35.0, 50.0, 60.0, 75.0, 90.0], 2)
            soc = cc[2].slider("Current SoC %", 5, 60, 20)
            dep = cc[3].slider("Departure slot", slot + 3, scn.config.n_slots, min(slot + 8, scn.config.n_slots))
            events = [replace(base, battery_kwh=kwh, soc_pct=float(soc), departure_slot=dep)]
        elif kind == "Departure time change":
            cand = [e.ev_id for e in scn.evs if state.assignments.get(e.ev_id) is None
                    or state.assignments[e.ev_id].start >= slot]
            if cand:
                ev_id = st.selectbox("EV", cand, format_func=lambda i: f"{scn.evs[i].name} "
                                     f"(departs {clock(scn, scn.evs[i].departure_slot)})")
                nd = st.slider("New departure slot", slot + 1, scn.config.n_slots,
                               max(slot + 1, scn.evs[ev_id].departure_slot - 3))
                events = [DepartureChange(slot, ev_id, nd)]
            else:
                err = "No EV has a session that is still to start at that slot."
        elif kind == "Station power reduction":
            sid = st.selectbox("Station", list(stn), format_func=stn.get)
            pct = st.slider("Power reduction %", 10, 80, 40)
            events = [PowerReduction(slot, sid, 1 - pct / 100)]
        elif kind == "Tariff change":
            cc = st.columns(3)
            a = cc[0].slider("From slot", slot, scn.config.n_slots - 1, slot + 2)
            b = cc[1].slider("To slot (exclusive)", a + 1, scn.config.n_slots, min(a + 4, scn.config.n_slots))
            mult = cc[2].slider("Price multiplier", 0.3, 3.0, 0.5, 0.1)
            events = [TariffChange(slot, a, b, mult)]
        else:
            run_ev = in_progress(state, slot)
            if run_ev:
                ev_id = st.selectbox("EV in progress", list(run_ev), format_func=lambda i: scn.evs[i].name)
                events = [SessionOverrun(slot, ev_id, st.slider("Extra slots", 1, 4, 2))]
            else:
                err = "No session is in progress at that slot."
    if err:
        st.warning(err)
    c1, c2 = st.columns([1, 1])
    if c1.button("⚡ Trigger event", type="primary", disabled=not events, width="stretch"):
        try:
            st.session_state["run"] = sim.trigger(events)
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.pop("run", None)
    if c2.button("↺ Reset to initial network", width="stretch"):
        sim.state = st.session_state["base_state"]
        st.session_state.pop("run", None)
        st.rerun()

    drun = st.session_state.get("run")
    if drun is not None:
        before, after, rep = drun.before, drun.after, drun.report
        for ev in drun.events:
            st.markdown(f'<div class="event">{ev.describe(after)}</div>', unsafe_allow_html=True)
        for n in drun.notes:
            st.caption("• " + n)
        c = st.columns(5)
        c[0].metric("EVs affected", len(rep.affected))
        c[1].metric("Reservations invalidated", len(rep.invalidated))
        c[2].metric("Sessions interrupted", len(after.interrupted))
        c[3].metric("Frozen (already started)", len(rep.frozen))
        c[4].metric("Untouched reservations", len([i for i in after.assignments
                                                  if i not in rep.affected and i not in rep.frozen]))
        if rep.unresolvable:
            st.error("Unavoidable overload from sessions that already started: " + "; ".join(rep.unresolvable))
        if rep.affected:
            st.dataframe(pd.DataFrame([{
                "EV": after.scn.evs[i].name, "Commitment": after.commitments[i].ctype.value,
                "Was": (f"{before.scn.stations[before.assignments[i].station_id].name} @ "
                        f"{clock(before.scn, before.assignments[i].start)}"
                        if i in before.assignments else "no reservation"),
                "Why affected": "; ".join(rep.reasons(i))} for i in sorted(rep.affected)]),
                hide_index=True, width="stretch")
        else:
            st.info("No reservation is affected: nothing to recover.")
        dn = down_map(before, after)
        st.pyplot(rviz.fig_gantt(before.scn, before.assignments, {}, before.commitments,
                                 "Original schedule", None))
        st.subheader("Recovery")
        chosen = st.multiselect("Strategies to run (all receive identical inputs)", list(STRATEGIES),
                                default=list(STRATEGIES))
        if st.button("▶ Run recovery", type="primary", disabled=not chosen):
            with st.spinner("Recovering the schedule (QAOA circuits are simulated on Aer)…"):
                drun.results.clear()
                drun.metrics.clear()
                sim.recover(drun, tuple(chosen), qcfg=qcfg, hcfg=hcfg)
        if drun.results:
            show = st.selectbox("Show recovered schedule of", list(drun.results),
                                index=list(drun.results).index("CP-SAT recovery")
                                if "CP-SAT recovery" in drun.results else 0)
            res_ = drun.results[show]
            mm = drun.metrics[show]
            moved = {i for i, o in res_.assignments.items() if i in before.assignments and
                     (o.station_id, o.start) != (before.assignments[i].station_id, before.assignments[i].start)}
            newset = {i for i in res_.assignments if i not in before.assignments}
            st.pyplot(rviz.fig_gantt(after.scn, res_.assignments, {}, after.commitments,
                                     f"After recovery: {show}", after.t_now, dn, moved, newset))
            st.caption("White outline = reservation changed · dashed = EV without an earlier reservation · "
                       "magenta line = event time · red hatch = station out of service.")
            if mm["valid"]:
                st.success("Independent validator: no hard rule broken in the recovered schedule.")
            else:
                st.error("Validator found hard violations:\n" + "\n".join(mm["hard_violations"]))
            c = st.columns(5)
            c[0].metric("Reassigned", mm["reassigned_reservations"])
            c[1].metric("Commitment violations", mm["commitment_violations"],
                        f"{mm['commitment_violations_forced']} forced · "
                        f"{mm['commitment_violations_avoidable']} avoidable", delta_color="off")
            c[2].metric("Departure SLA", f"{mm['sla_success_pct']:.0f}%")
            c[3].metric("Unassigned EVs", mm["unassigned_evs"])
            c[4].metric("Recovery time", f"{mm['recovery_runtime_s'] * 1000:.0f} ms")
            if mm["unassigned_evs"]:
                nopt = set(res_.meta.get("unavoidable_no_option", []))
                st.warning("Unserved EVs are never hidden: " + ", ".join(
                    f"{n} ({'no physically feasible slot' if after.scn.evs[i].name in mm['unserved_evs'] and i in nopt else 'no free capacity before departure'})"
                    for i, n in ((e.ev_id, e.name) for e in after.scn.evs if e.name in mm["unserved_evs"])))
            diff = []
            for ev in after.scn.evs:
                i = ev.ev_id
                o, n = before.assignments.get(i), res_.assignments.get(i)
                f = lambda x: "-" if x is None else f"{after.scn.stations[x.station_id].name} {clock(after.scn, x.start)}"
                kind_ = ("kept" if o and n and (o.station_id, o.start) == (n.station_id, n.start)
                         else "moved" if o and n else "DROPPED" if o else "newly served" if n else "unserved")
                if kind_ != "kept" or i in res_.free_ids:
                    diff.append({"EV": ev.name, "Commitment": after.commitments[i].ctype.value,
                                 "Before": f(o), "After": f(n), "Change": kind_})
            if diff:
                st.dataframe(pd.DataFrame(diff), hide_index=True, width="stretch")
            if res_.stages:
                st.caption(f"The search area grew {res_.stages} time(s) beyond the directly affected EVs "
                           f"because some EV could not be placed otherwise.")
            if "Hybrid" in show and res_.meta.get("fallback_batches"):
                st.info(f"{res_.meta['fallback_batches']} QAOA batch(es) used the classical fallback.")
            if st.button("Adopt this schedule as the current network"):
                sim.commit(drun, show)
                st.session_state.pop("run", None)
                st.rerun()

# ------------------------------------------------------------------ 4. recovery comparison
with tabs[3]:
    drun = st.session_state.get("run")
    if drun is None or not drun.metrics:
        st.info("Trigger a disruption and run recovery in the **Disruption simulator** tab first.")
    else:
        rows = list(drun.metrics.values())
        valid_rows = [r for r in rows if r["valid"]]
        best_r = min(valid_rows, key=lambda r: r["recovery_objective"])
        ties = [r["strategy"] for r in valid_rows if abs(r["recovery_objective"] - best_r["recovery_objective"]) < 1e-6]
        st.markdown(f"**In short:** all strategies saw the identical network, schedule and event. Lowest recovery "
                    f"objective: **{' = '.join(ties)}**. Ties mean the problem is small enough for a simple method "
                    "to reach the same plan; the quantum-assisted result is reported as-is, with no advantage claimed.")
        frame = metrics_frame(rows)
        frame.columns = ["Runtime (s)", "Cost (₹)", "Avg wait (min)", "Max wait (min)", "Reassigned",
                         "Commit. violations", "SLA %", "Critical SLA %", "Unassigned EVs",
                         "Unserved kWh", "Utilization %", "Grid violations", "Fairness (Jain)",
                         "Recovery objective", "Valid"]
        st.dataframe(frame.round(2), width="stretch")
        st.pyplot(rviz.fig_strategy_bars(rows))
        with st.expander("Recovery objective by term (lower is better)"):
            st.dataframe(pd.DataFrame({r["strategy"]: r["objective_terms"] for r in rows}).T.round(3),
                         width="stretch")
            st.caption("operational = energy + waiting + travel · reassignment and commitment = penalties for moving "
                       "reservations · missed_deadline = penalty for EVs left uncharged (critical EVs cost 1000).")
        with st.expander("Commitment violations: forced vs avoidable"):
            st.dataframe(pd.DataFrame({r["strategy"]: {
                "forced by the disruption": r["commitment_violations_forced"],
                "avoidable (valid reservation moved)": r["commitment_violations_avoidable"],
                "moved": r["moved"], "dropped": r["dropped"],
                "unserved energy (kWh)": round(r["unserved_energy_kwh"], 1)} for r in rows}).T,
                width="stretch")
        grid_sid = next((e.station_id for e in drun.events if isinstance(e, PowerReduction)), None)
        if grid_sid is not None:
            pick = st.selectbox("Grid power at the reduced station", list(drun.metrics), key="gridpick")
            kw = np.array(drun.metrics[pick]["evaluator"]["occupancy_kw"])
            st.pyplot(rviz.fig_station_power(drun.after.scn, kw, grid_sid,
                                             f"{drun.after.scn.stations[grid_sid].name}: charging power vs limit ({pick})"))
    st.subheader("Multi-seed stress benchmark")
    st.caption("Same scenarios and inputs for every strategy, repeated over seeds. One run is an anecdote; this is the spread.")
    bc = st.columns(3)
    sc_pick = bc[0].multiselect("Scenarios", list(SCENARIOS), default=["failure", "arrivals", "combined"],
                                format_func=lambda k: SCENARIOS[k].title)
    n_seeds = bc[1].slider("Seeds", 2, 10, 4)
    bn = bc[2].selectbox("EVs per network", [12, 16], 0)
    inc_h = st.checkbox("Include hybrid QAOA (slower)", False)
    if st.button("Run benchmark", disabled=not sc_pick):
        strat = tuple(s for s in STRATEGIES if inc_h or not s.startswith("Hybrid"))
        with st.spinner("Running…"):
            bdf = run_benchmark(sc_pick, tuple(range(1, n_seeds + 1)), bn, strat, qcfg)
        st.session_state["bench"] = bdf
    if "bench" in st.session_state:
        bdf = st.session_state["bench"]
        st.dataframe(summarize(bdf).round(2), width="stretch")
        st.caption(f"{bdf['valid'].mean() * 100:.0f}% of all recovered schedules passed the independent validator.")
        piv = bdf.pivot_table(index="scenario", columns="strategy", values="recovery_objective")
        st.bar_chart(piv.drop(columns=[c for c in piv if c.startswith("Static")], errors="ignore"))

# ------------------------------------------------------------------ 5. quantum engine
with tabs[4]:
    qr = red["qaoa"]
    s = qaoa_summary(qr)
    q = red["qubo"]
    EXACT = "Exact QUBO optimum (brute force)"
    rc = red["random_control"]
    st.markdown("**What runs where:** preprocessing, option generation, batching, decoding, local refinement and "
                "validation run on a normal CPU. Only the *sampling of QUBO bitstrings* is a quantum-circuit "
                "**simulation** (Qiskit Aer on this machine, not real hardware). The parameter tuning is a classical "
                "optimizer (SciPy).")
    opt = red["objective_by_method"][EXACT]
    got = s["best_feasible_cost"]
    st.markdown(
        f"**Reduced instance** ({len(red['ev_ids'])} EVs, {s['qubits']} qubits): the circuit "
        f"{'found the exact optimum' if red['qaoa_matches_exact'] else 'did not find the exact optimum'}"
        f"{'' if got is None else f' (cost {got:.3f} vs optimum {opt:.3f})'}. Random sampling with the same "
        "shot budget reaches it too, so **no quantum advantage is claimed** at this size.")
    c = st.columns(6)
    c[0].metric("Qubits", s["qubits"])
    c[1].metric("Decision / slack", f"{red['n_decision']} / {red['n_slack']}")
    c[2].metric("Circuit depth", s["circuit_depth"])
    c[3].metric("Gates", s["gate_total"])
    c[4].metric("Shots × evals", f"{s['shots']} × {s['evaluations']}")
    c[5].metric("Sim time", f"{s['sim_time_s']:.1f} s")
    l, r_ = st.columns(2)
    with l:
        st.subheader("QUBO formulation")
        st.latex(r"E(x)=\sum_o c_o x_o+P_a\sum_v\Big(\sum_{o\in v}x_o-1\Big)^2"
                 r"+P_c\sum_{s,t}\Big(\sum_{o\ni(s,t)}x_o+\sum_k w_k y_k-r_{s,t}\Big)^2"
                 r"+P_c\!\!\sum_{p_o+p_{o'}>G_{s,t}}\!\!x_o x_{o'}")
        st.caption("c_o is the normalized cost. In recovery it also contains λ₁·[moved] + λ₂(commitment)·[moved] "
                   "(+ fairness), and non-critical EVs get an explicit 'stay unserved' variable priced at its penalty. "
                   "The last term enforces the aggregate grid power limit pairwise; every decoded sample is also "
                   "checked exactly, so a ≥3-way power overload is never reported feasible.")
        st.subheader("QUBO → Ising")
        st.latex(r"x_i=\tfrac{1-z_i}{2}\;\Rightarrow\;H_C=\text{const}+\sum_i h_iZ_i+\sum_{i<j}J_{ij}Z_iZ_j")
        h_, J_, c0 = to_ising(q)
        st.caption(f"{len(h_)} fields h_i, {len(J_)} couplings J_ij, constant {c0:.2f}. Equality with the QUBO is "
                   "unit-tested on random bitstrings.")
    with r_:
        st.subheader("QAOA circuit")
        st.caption(f"H on every qubit, then {s['depth']}× [RZ(2γh) · RZZ(2γJ) · RX(2β)], then measurement. "
                   f"Gate counts after transpilation: {s['gate_counts']}")
        st.caption("The full diagram is wide; open it below.")
    with st.expander("Show circuit diagram (reduced instance)"):
        try:
            qc, _, _ = build_circuit(q, s["depth"])
            st.pyplot(qc.draw("mpl", fold=60, style="iqp-dark"))
        except Exception as exc:                                    # drawing must never break the tab
            st.caption(f"Circuit drawing unavailable: {exc}")
    l, r_ = st.columns(2)
    with l:
        st.subheader("Measurement distribution")
        counts = qr.final_counts
        info = red["bit_info"]
        st.pyplot(viz.fig_distribution(counts, {k: info[k][0] for k in counts}, {k: info[k][1] for k in counts}))
    with r_:
        st.subheader("Optimizer convergence")
        ex = red["exact"]["min_energy"] if red["exact"] else None
        st.pyplot(viz.fig_qaoa_progress(qr.history, ex))
    st.subheader("QAOA vs random sampling vs exact vs CP-SAT (same options)")
    cmp = pd.DataFrame({"Chance a sample is the best plan (%)": {
        "QAOA": 100 * s["prob_best_feasible_final"], "Random sampling": 100 * rc.get("prob_best", 0.0)},
        "Chance a sample is valid (%)": {"QAOA": 100 * s["prob_feasible_final"],
                                         "Random sampling": 100 * rc["prob_feasible"]}})
    c = st.columns(2)
    c[0].bar_chart(cmp)
    c[1].dataframe(pd.DataFrame({"objective (lower = better)": red["objective_by_method"]}).dropna(),
                   width="stretch")
    if res["depth_sweep"]:
        st.dataframe(pd.DataFrame(res["depth_sweep"]), hide_index=True, width="stretch")
    st.subheader("Hybrid batches")
    st.caption("Initial schedule (rolling horizon):")
    st.dataframe(pd.DataFrame([{
        "batch": l_.batch, "EVs": ", ".join(scn0.evs[i].name for i in l_.ev_ids), "qubits": l_.n_qubits,
        "decision vars": l_.n_decision, "slack vars": l_.n_slack,
        "circuit depth": l_.qaoa.circuit_depth if l_.qaoa else None,
        "evaluations": l_.qaoa.n_evaluations if l_.qaoa else None,
        "valid samples %": round(100 * l_.qaoa.prob_feasible, 1) if l_.qaoa else None,
        "used classical fallback": l_.used_fallback} for l_ in res["hybrid_logs"]]),
        hide_index=True, width="stretch")
    drun = st.session_state.get("run")
    hres = drun.results.get("Hybrid QAOA recovery") if drun else None
    if hres is not None:
        st.caption("Recovery (last disruption):")
        logs = [l_ for l_ in hres.logs]
        st.dataframe(pd.DataFrame([{
            "batch": l_.batch, "EVs": ", ".join(drun.after.scn.evs[i].name for i in l_.ev_ids),
            "qubits": l_.n_qubits, "decision vars": l_.n_decision, "slack vars": l_.n_slack,
            "circuit depth": l_.qaoa.circuit_depth if l_.qaoa else None,
            "gates": sum(l_.qaoa.gate_counts.values()) if l_.qaoa else None,
            "shots": l_.qaoa.config.shots if l_.qaoa else None,
            "evaluations": l_.qaoa.n_evaluations if l_.qaoa else None,
            "sim time (s)": round(l_.qaoa.sim_time_s, 2) if l_.qaoa else None,
            "valid samples %": round(100 * l_.qaoa.prob_feasible, 1) if l_.qaoa else None,
            "best measured energy": round(l_.qaoa.best_energy, 3) if l_.qaoa else None,
            "classical fallback": l_.used_fallback, "note": l_.note} for l_ in logs]),
            hide_index=True, width="stretch")
    else:
        st.caption("Run hybrid recovery in the Disruption simulator to see its QAOA batches here.")

# ------------------------------------------------------------------ 6. benchmarks
with tabs[5]:
    cl = dict(evals)
    cheapest = min(cl, key=lambda n: cl[n]["total_cost_inr"] if cl[n]["unassigned_count"] == 0 else 1e18)
    st.markdown(f"**Initial scheduling, no disruption.** The exact classical solver (OR-Tools) is hard to beat at "
                f"this size. Lowest total cost among plans that charge everyone: **{cheapest}** "
                f"(₹{cl[cheapest]['total_cost_inr']:,.0f}). The hybrid plan uses the quantum circuit for each "
                "small batch; we do not claim it is better.")
    tbl = pd.DataFrame({n: {
        "Valid?": "yes" if m["feasible"] else "no",
        "Total cost (₹)": round(m["total_cost_inr"]),
        "Electricity (₹)": round(m["energy_cost_inr"]),
        "Waiting (₹)": round(m["waiting_cost_inr"]),
        "Travel (₹)": round(m["travel_cost_inr"]),
        "Avg wait (min)": round(m["avg_wait_h"] * 60),
        "EVs left uncharged": m["unassigned_count"],
        "Time to compute (s)": round(m["runtime_s"], 2)} for n, m in cl.items()}).T
    st.dataframe(tbl, width="stretch")
    st.pyplot(viz.fig_comparison(cl, "All plans side by side"))
    csv = Path(__file__).parent / "outputs" / "scaling.csv"
    st.subheader("Bigger datasets")
    if not csv.exists():
        st.info("Run `python run_scaling.py` to measure how the methods behave as the dataset grows.")
    else:
        df = pd.read_csv(csv)
        top = df["evs"].max()
        g = df[df["evs"] == top].set_index("method")
        st.markdown(
            f"We grew the problem from {df['evs'].min()} to {top} EVs and timed every method. "
            f"At {top} EVs the exact solver (OR-Tools) took {g.loc['OR-Tools', 'runtime_s']:.0f} s"
            f"{' and hit its time limit without proving the best plan' if g.loc['OR-Tools', 'cpsat_status'] == 'FEASIBLE' else ''}; "
            f"the quantum-batch plan took {g.loc['Hybrid (QAOA batches)', 'runtime_s']:.0f} s.")
        st.line_chart(df.pivot(index="evs", columns="method", values="objective"))
        st.line_chart(df.pivot(index="evs", columns="method", values="runtime_s"))
        st.caption("Simulated quantum time is on a laptop simulator, not real hardware, "
                   "so time comparisons say nothing about real quantum speed.")
