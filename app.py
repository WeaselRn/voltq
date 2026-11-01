"""VoltQ - charging network operations dashboard (single page).  Run: streamlit run app.py

Planning is done by the hybrid quantum-classical optimizer (QAOA simulated on Qiskit Aer + classical
refinement). Incidents are simulated; no live charger is contacted.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from voltq.data import generate_scenario
from voltq.evaluate import evaluate_sessions
from voltq.hybrid import HybridConfig, hybrid_schedule
from voltq.models import ScenarioConfig, assign_ports, grid_limit_kw
from voltq.preprocess import build_options
from voltq.qaoa import QAOAConfig
from voltq.recovery import viz as rviz
from voltq.recovery.commitments import RecoveryWeights, assign_commitments
from voltq.recovery.events import (DepartureChange, PowerReduction, SessionOverrun, StationOutage,
                                   TariffChange, in_progress)
from voltq.recovery.state import NetworkState
from voltq.simulation.network_simulator import NetworkSimulator
from voltq.simulation.scenarios import EVENT_SLOT, arrival_near

STRATEGY = "Hybrid QAOA recovery"

st.set_page_config(page_title="VoltQ", page_icon="⚡", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1rem;max-width:1500px}
header[data-testid=stHeader]{background:transparent}
[data-testid=stMetric]{background:#16202c;border:1px solid #2a3646;border-radius:12px;padding:12px 16px}
[data-testid=stMetricLabel]{color:#8b98a9}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.75rem;font-weight:700}
.ok{background:#14532d;color:#86efac}.warn{background:#713f12;color:#fde68a}.bad{background:#7f1d1d;color:#fecaca}
.card{background:#16202c;border:1px solid #2a3646;border-radius:12px;padding:12px 14px;margin-bottom:10px}
.card h4{margin:0 0 4px 0;font-size:1rem}.muted{color:#8b98a9;font-size:.82rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.alert{border-left:4px solid #e879f9;background:#16202c;padding:8px 12px;border-radius:6px;margin-bottom:6px;font-size:.88rem}
.alert.good{border-color:#22c55e}.alert.bad{border-color:#ef4444}
.brand{font-size:1.7rem;font-weight:800;letter-spacing:.5px}.brand span{color:#2dd4bf}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ network setup
with st.sidebar:
    st.header("Network setup")
    n_evs = st.slider("EVs", 6, 40, 12)
    n_st = st.slider("Stations", 2, 10, 4)
    seed = st.number_input("Seed", 0, 10_000, 42)
    depth = st.slider("Optimizer depth (QAOA layers)", 1, 3, 1)
    rebuild = st.button("Rebuild network", type="primary", use_container_width=True)


@st.cache_resource(show_spinner="Planning the charging network…")
def build(n_evs: int, n_st: int, seed: int, depth: int):
    cfg = ScenarioConfig(n_evs=n_evs, n_stations=n_st, seed=seed)
    qcfg = QAOAConfig(depth=depth, shots=256, maxiter=40, seed=seed)
    scn = generate_scenario(cfg)
    pre = build_options(scn)
    sched = hybrid_schedule(scn, pre, qcfg, HybridConfig())[1]
    commit = assign_commitments(scn, pre, sched.assignments, seed)
    return NetworkState(scn, pre.refs, dict(sched.assignments), commit, 0, weights=RecoveryWeights()), qcfg


key = (n_evs, n_st, int(seed), depth)
if st.session_state.get("key") != key or rebuild:
    base, qcfg_ = build(*key)
    st.session_state.update(key=key, sim=NetworkSimulator(base.copy()), base=base, qcfg=qcfg_, last=None)
sim: NetworkSimulator = st.session_state["sim"]
qcfg: QAOAConfig = st.session_state["qcfg"]
state = sim.state
scn = state.scn


def clock(slot: int) -> str:
    return rviz.slot_labels(scn)[min(slot, scn.config.n_slots)]


# ------------------------------------------------------------------ derived status
now = state.t_now
status, busy = {}, {}
for s in scn.stations:
    down = bool(state.event_log) and s.close_slot <= now and s.close_slot < scn.config.n_slots
    status[s.station_id] = "down" if down else "limited" if s.grid_kw is not None else "online"
    busy[s.station_id] = sum(1 for o in state.assignments.values()
                             if o.station_id == s.station_id and o.start <= now < o.start + o.duration)
unassigned = [e for e in scn.evs if e.ev_id not in state.assignments]
online = [s for s in scn.stations if status[s.station_id] != "down"]

# ------------------------------------------------------------------ header
h1, h2 = st.columns([3, 2])
h1.markdown('<div class="brand">⚡ VOLT<span>Q</span> &nbsp;<span class="muted">Charging network control</span></div>',
            unsafe_allow_html=True)
pill = ('<span class="pill bad">DEGRADED</span>' if "down" in status.values() or unassigned
        else '<span class="pill warn">LIMITED</span>' if "limited" in status.values()
        else '<span class="pill ok">ALL SYSTEMS NORMAL</span>')
h2.markdown(f"<div style='text-align:right;padding-top:.6rem'>{pill} &nbsp; "
            f"<span class='muted'>Now {clock(now)} · slot {now}</span></div>", unsafe_allow_html=True)

c = st.columns(5)
c[0].metric("Active reservations", len(state.assignments))
c[1].metric("Stations online", f"{len(online)} / {len(scn.stations)}")
c[2].metric("Free ports now", sum(s.ports - busy[s.station_id] for s in online))
c[3].metric("Grid power available", f"{sum(grid_limit_kw(s, now) for s in online):.0f} kW")
c[4].metric("Waiting for a slot", len(unassigned))

# ------------------------------------------------------------------ main row
left, right = st.columns([2, 1], gap="large")
last = st.session_state.get("last")

with left:
    st.subheader("Network map")
    st.pyplot(rviz.fig_map(scn, status, busy))
    moved = set()
    if last is not None:
        moved = {i for i, o in state.assignments.items() if i in last.before.assignments and
                 (o.station_id, o.start) != (last.before.assignments[i].station_id,
                                             last.before.assignments[i].start)}
    st.subheader("Charging schedule")
    down_map = {s.station_id: s.close_slot for s in scn.stations if status[s.station_id] == "down"}
    st.pyplot(rviz.fig_gantt(scn, state.assignments, {}, state.commitments, "",
                             now if state.event_log else None, down_map, moved, None))
    st.caption("Colour = how firmly the booking is promised (red critical, green confirmed, amber flexible, "
               "blue new). White outline = booking changed by the last re-plan.")

with right:
    st.subheader("Report an incident")
    kind = st.selectbox("What happened?", ["Station out of service", "Station power limited",
                                           "New EV arrives", "Driver leaves earlier",
                                           "Electricity price change", "Charger overrun"],
                        label_visibility="collapsed")
    slot = st.slider("At slot", 1, scn.config.n_slots - 2, min(EVENT_SLOT, scn.config.n_slots - 2),
                     help="Time slot (30 min each) when it happens")
    stn = {s.station_id: s.name for s in scn.stations}
    events, err = [], None
    if kind == "Station out of service":
        events = [StationOutage(slot, st.selectbox("Station", list(stn), format_func=stn.get))]
    elif kind == "Station power limited":
        sid = st.selectbox("Station", list(stn), format_func=stn.get)
        events = [PowerReduction(slot, sid, 1 - st.slider("Power cut %", 10, 80, 40) / 100)]
    elif kind == "New EV arrives":
        near = st.selectbox("Arrives near", list(stn), format_func=stn.get)
        base_ev = arrival_near(state, near, slot, 0, int(seed))
        soc = st.slider("Battery now %", 5, 60, 20)
        stay = st.slider("Stays (slots)", 3, 12, 8)
        events = [replace(base_ev, soc_pct=float(soc), departure_slot=min(slot + stay, scn.config.n_slots))]
    elif kind == "Driver leaves earlier":
        cand = [e.ev_id for e in scn.evs if state.assignments.get(e.ev_id) is None
                or state.assignments[e.ev_id].start >= slot]
        if cand:
            i = st.selectbox("Driver", cand, format_func=lambda i: f"{scn.evs[i].name} "
                             f"(leaves {clock(scn.evs[i].departure_slot)})")
            events = [DepartureChange(slot, i, st.slider("New departure slot", slot + 1, scn.config.n_slots,
                                                         max(slot + 1, scn.evs[i].departure_slot - 3)))]
        else:
            err = "No booking left to change at that time."
    elif kind == "Electricity price change":
        a = st.slider("From slot", slot, scn.config.n_slots - 1, min(slot + 2, scn.config.n_slots - 1))
        b = st.slider("Until slot", a + 1, scn.config.n_slots, min(a + 4, scn.config.n_slots))
        events = [TariffChange(slot, a, b, st.slider("Price ×", 0.3, 3.0, 0.5, 0.1))]
    else:
        run_ev = in_progress(state, slot)
        if run_ev:
            i = st.selectbox("Vehicle", list(run_ev), format_func=lambda i: scn.evs[i].name)
            events = [SessionOverrun(slot, i, st.slider("Extra slots", 1, 4, 2))]
        else:
            err = "No vehicle is charging at that time."
    if err:
        st.warning(err)
    if st.button("Report and re-plan", type="primary", disabled=not events, use_container_width=True):
        try:
            with st.spinner("Re-planning with the hybrid optimizer…"):
                run = sim.trigger(events)
                sim.recover(run, (STRATEGY,), qcfg=qcfg, hcfg=HybridConfig())
                st.session_state["last"] = run
                sim.commit(run, STRATEGY)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if st.button("Reset network", use_container_width=True):
        st.session_state["sim"] = NetworkSimulator(st.session_state["base"].copy())
        st.session_state["last"] = None
        st.rerun()

    st.subheader("Alerts")
    if last is None:
        st.markdown('<div class="alert good">No incidents. All reservations are on plan.</div>',
                    unsafe_allow_html=True)
    else:
        m = last.metrics[STRATEGY]
        for ev in last.events:
            st.markdown(f'<div class="alert bad">{ev.describe(last.after)}</div>', unsafe_allow_html=True)
        for n in last.notes:
            st.markdown(f'<div class="alert">{n}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="alert good"><b>Re-planned in {m["recovery_runtime_s"]:.1f} s.</b> '
                    f'{len(last.report.affected)} EVs affected, {m["reassigned_reservations"]} bookings changed, '
                    f'{m["sla_success_pct"]:.0f}% of drivers charged on time.</div>', unsafe_allow_html=True)
        if m["valid"]:
            st.markdown('<div class="alert good">Safety check passed: no port, power or time rule broken.</div>',
                        unsafe_allow_html=True)
        else:
            st.error("Safety check failed:\n" + "\n".join(m["hard_violations"]))
        if m["unassigned_evs"]:
            st.markdown(f'<div class="alert bad">Cannot be charged in time: {", ".join(m["unserved_evs"])}</div>',
                        unsafe_allow_html=True)
        res = last.results[STRATEGY]
        rows = []
        for ev in last.after.scn.evs:
            o, n = last.before.assignments.get(ev.ev_id), res.assignments.get(ev.ev_id)
            if o is None and n is None:
                continue
            if o and n and (o.station_id, o.start) == (n.station_id, n.start):
                continue
            f = lambda x: "-" if x is None else f"{scn.stations[x.station_id].name} {clock(x.start)}"
            rows.append({"EV": ev.name, "Was": f(o), "Now": f(n)})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ stations + reservations
st.subheader("Stations")
cards = []
for s in scn.stations:
    k = status[s.station_id]
    pill_ = {"online": '<span class="pill ok">ONLINE</span>', "limited": '<span class="pill warn">LIMITED</span>',
             "down": '<span class="pill bad">OFFLINE</span>'}[k]
    kw = "0" if k == "down" else f"{grid_limit_kw(s, now):.0f}"
    cards.append(f'<div class="card"><h4>{s.name} {pill_}</h4>'
                 f'<div class="muted">{busy[s.station_id]}/{s.ports} ports busy · {kw} kW available · '
                 f'{s.power_kw:.0f} kW per port</div></div>')
st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)

st.subheader("Reservations")
ports = {x.ev_id: x.port for x in assign_ports(state.assignments, scn.stations)}
rows = []
for ev in scn.evs:
    o = state.assignments.get(ev.ev_id)
    rows.append({"EV": ev.name, "Promise": state.commitments[ev.ev_id].ctype.value,
                 "Station": scn.stations[o.station_id].name if o else "-",
                 "Port": ports[ev.ev_id] + 1 if o else None,
                 "From": clock(o.start) if o else "-", "To": clock(o.start + o.duration) if o else "-",
                 "Energy (kWh)": round(o.energy_kwh, 1) if o else None,
                 "Leaves": clock(ev.departure_slot), "Status": "booked" if o else "WAITING"})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption("Planned by the hybrid quantum-classical optimizer (QAOA simulated on a local computer, "
           "classical refinement, independent safety check). Incidents are simulated.")
