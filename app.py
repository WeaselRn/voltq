"""VoltQ dashboard.  Run: streamlit run app.py   (all numbers come from the optimization engine)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from voltq import viz
from voltq.hybrid import HybridConfig
from voltq.models import ScenarioConfig
from voltq.pipeline import qaoa_summary, schedule_table
from voltq.qaoa import OPTIMIZERS, QAOAConfig
from voltq.study import best_method, run_all

st.set_page_config(page_title="VoltQ", page_icon="⚡", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem}
[data-testid=stMetric]{background:#f4f6fb;border-radius:10px;padding:10px 14px}
@media (prefers-color-scheme: dark){[data-testid=stMetric]{background:#1c2230}}
</style>""", unsafe_allow_html=True)

st.title("⚡ VoltQ")
st.caption("Quantum-assisted EV charging optimization · QAOA runs locally on Qiskit Aer · "
           "no IBM hardware or cloud")

with st.sidebar:
    st.header("Scenario")
    n_evs = st.slider("EVs", 4, 15, 10)
    n_st = st.slider("Stations", 2, 5, 4)
    pmin, pmax = st.slider("Ports per station (min, max)", 1, 3, (2, 2))
    n_slots = st.slider("Time slots (30 min)", 8, 16, 12)
    seed = st.number_input("Random seed", 0, 10_000, 42)
    st.subheader("Electricity price (EUR/kWh)")
    peak = st.number_input("Peak", 0.05, 1.0, 0.42, 0.01)
    normal = st.number_input("Normal", 0.05, 1.0, 0.26, 0.01)
    off = st.number_input("Off-peak", 0.02, 1.0, 0.13, 0.01)
    start_hour = st.slider("Window start (hour of day)", 0.0, 23.5, 19.0, 0.5)
    st.subheader("Objective weights")
    alpha = st.slider("alpha (energy)", 0.0, 3.0, 1.0, 0.1)
    beta = st.slider("beta (waiting)", 0.0, 3.0, 1.0, 0.1)
    gamma = st.slider("gamma (travel)", 0.0, 3.0, 1.0, 0.1)
    st.header("QAOA / hybrid")
    depth = st.slider("QAOA depth p", 1, 3, 1)
    shots = st.select_slider("Shots", [128, 256, 512, 1024], 512)
    optimizer = st.selectbox("Classical optimizer", OPTIMIZERS)
    maxiter = st.slider("Max iterations", 10, 120, 60, 10)
    cvar = st.slider("CVaR alpha (1 = mean)", 0.1, 1.0, 1.0, 0.1)
    batch = st.slider("Hybrid batch size (EVs)", 2, 4, 3)
    sweep = st.checkbox("Run depth sweep p=1..3 (slower)", False)
    run = st.button("Run optimization", type="primary", use_container_width=True)


@st.cache_resource(show_spinner="Optimizing (Aer sampling)…")
def cached_run(cfg_t, q_t, b, sweep_flag):
    cfg = ScenarioConfig(**dict(cfg_t))
    qcfg = QAOAConfig(**dict(q_t))
    return run_all(cfg, qcfg, HybridConfig(batch_size=b), do_sweep=sweep_flag)


if run or "last" not in st.session_state:
    cfg_t = tuple(dict(n_evs=n_evs, n_stations=n_st, ports_min=pmin, ports_max=pmax, n_slots=n_slots,
                       seed=int(seed), peak_price=peak, normal_price=normal, offpeak_price=off,
                       start_hour=start_hour, alpha=alpha, beta=beta, gamma=gamma).items())
    q_t = tuple(dict(depth=depth, shots=shots, optimizer=optimizer, maxiter=maxiter,
                     seed=int(seed), cvar_alpha=cvar).items())
    st.session_state["last"] = cached_run(cfg_t, q_t, batch, sweep)
res = st.session_state["last"]
scn, pre, evals, scheds, red = res["scenario"], res["pre"], res["evals"], res["schedules"], res["reduced"]
best = best_method(evals)
qr = red["qaoa"]

tabs = st.tabs(["Overview", "Schedule", "Stations", "Quantum", "Comparison", "Visuals"])

with tabs[0]:
    demand = sum(o[0].energy_kwh for o in pre.options.values() if o)
    c = st.columns(5)
    c[0].metric("EVs", len(scn.evs))
    c[1].metric("Stations", len(scn.stations))
    c[2].metric("Energy demand", f"{demand:.0f} kWh")
    c[3].metric("Binary vars (naive → reduced)", f"{pre.full_size} → {pre.reduced_size}")
    c[4].metric("Best validated method", best)
    m = evals[best]
    c = st.columns(5)
    c[0].metric("Total cost", f"€{m['total_cost_eur']:.2f}")
    c[1].metric("Avg waiting", f"{m['avg_wait_h']:.2f} h")
    c[2].metric("Travel", f"{m['total_distance_km']:.1f} km")
    c[3].metric("Avg utilization", f"{m['avg_utilization_pct']:.1f}%")
    c[4].metric("EVs served", f"{m['completion_pct']:.0f}%")
    if pre.infeasible_evs:
        st.warning(f"No feasible option at all for: {[scn.evs[i].name for i in pre.infeasible_evs]}")
    unserved = [n for n, mm in evals.items() if mm["unassigned_count"]]
    if unserved:
        st.warning("Some methods leave EVs unassigned (reported, never silently violated): "
                   + ", ".join(f"{n} ({evals[n]['unassigned_count']})" for n in unserved))
    st.info("Quantum scope: QAOA samples only small reduced QUBOs (≤18 qubits) on Aer. The full problem "
            "is solved by batching + classical preprocessing/refinement. Feasibility is claimed only "
            "when the independent validator passes.")
    st.pyplot(viz.fig_tariff(scn))

with tabs[1]:
    pick = st.selectbox("Schedule", list(scheds), index=list(scheds).index(best))
    st.dataframe(schedule_table(scn, scheds[pick]), use_container_width=True, hide_index=True)
    if evals[pick]["violations"]:
        st.error("Validator violations:\n" + "\n".join(evals[pick]["violations"]))
    else:
        st.success("Validator: no constraint violations among assigned EVs.")
    st.pyplot(viz.fig_schedule(scn, scheds[pick].sessions(scn.stations), f"Charging timeline — {pick}"))
    ev_df = pd.DataFrame([e.__dict__ for e in scn.evs])
    with st.expander("EV and station input data"):
        st.dataframe(ev_df, hide_index=True)
        st.dataframe(pd.DataFrame([s.__dict__ for s in scn.stations]), hide_index=True)

with tabs[2]:
    pick2 = st.selectbox("Method", list(evals), index=list(evals).index(best), key="util")
    mu = evals[pick2]
    occ = np.array(mu["occupancy"])
    rows = [{"station": s.name, "ports": s.ports, "power kW": s.power_kw,
             "utilization %": round(mu["utilization_pct"][s.name], 1),
             "peak occupancy": int(occ[s.station_id].max()),
             "free port-slots": int(s.ports * (s.close_slot - s.open_slot) - occ[s.station_id].clip(max=s.ports).sum())}
            for s in scn.stations]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.pyplot(viz.fig_utilization(scn, mu, f"Station utilization — {pick2}"))

with tabs[3]:
    s = qaoa_summary(qr)
    st.subheader("Reduced-instance QAOA (Aer shot sampling)")
    c = st.columns(5)
    c[0].metric("Qubits", s["qubits"], f"{red['n_decision']} decision + {red['n_slack']} slack")
    c[1].metric("Circuit depth", s["circuit_depth"])
    c[2].metric("Gates", s["gate_total"])
    c[3].metric("Optimizer iterations", s["evaluations"])
    c[4].metric("Simulation time", f"{s['sim_time_s']:.2f} s")
    st.write(f"p={s['depth']}, shots={s['shots']}, optimizer={s['optimizer']}, seed={s['seed']}, "
             f"CVaR α={s['cvar_alpha']}. Gate counts: `{s['gate_counts']}`")
    c = st.columns(4)
    c[0].metric("Best measured objective", f"{s['best_feasible_cost']:.4f}" if s["best_feasible_cost"] is not None else "none")
    c[1].metric("Exact optimum", f"{red['objective_by_method']['Exact QUBO optimum (brute force)']:.4f}")
    c[2].metric("Best candidate feasible", "yes" if s["best_candidate_feasible"] else "NO")
    c[3].metric("P(feasible) final", f"{100 * s['prob_feasible_final']:.1f}%")
    rc = red["random_control"]
    st.caption(f"Control — uniform random sampling with the same budget ({rc['samples']} samples): "
               f"P(feasible)={100 * rc['prob_feasible']:.1f}%, best feasible cost={rc['best_feasible_cost']}. "
               "If random sampling matches QAOA, no quantum benefit is shown on this size.")
    ex = red["exact"]["min_energy"] if red["exact"] else None
    st.pyplot(viz.fig_qaoa_progress(qr.history, ex))
    bi = red["bit_info"]
    counts = qr.final_counts
    st.pyplot(viz.fig_distribution(counts, {k: bi[k][0] for k in counts}, {k: bi[k][1] for k in counts}))
    if res["depth_sweep"]:
        st.subheader("QAOA depth comparison")
        st.dataframe(pd.DataFrame(res["depth_sweep"]), hide_index=True, use_container_width=True)
    st.subheader("Hybrid batches (each QAOA call)")
    st.dataframe(pd.DataFrame([{
        "batch": l.batch, "EVs": [scn.evs[i].name for i in l.ev_ids], "qubits": l.n_qubits,
        "iterations": l.qaoa.n_evaluations if l.qaoa else None,
        "circuit depth": l.qaoa.circuit_depth if l.qaoa else None,
        "sim s": round(l.qaoa.sim_time_s, 2) if l.qaoa else None,
        "fallback to greedy": l.used_fallback, "note": l.note} for l in res["hybrid_logs"]]),
        hide_index=True, use_container_width=True)

with tabs[4]:
    tbl = pd.DataFrame({n: {
        "feasible (validated)": m["feasible"], "objective": round(m["objective"], 3),
        "total cost €": round(m["total_cost_eur"], 2), "energy €": round(m["energy_cost_eur"], 2),
        "waiting €": round(m["waiting_cost_eur"], 2), "travel €": round(m["travel_cost_eur"], 2),
        "avg wait h": round(m["avg_wait_h"], 2), "distance km": round(m["total_distance_km"], 1),
        "unassigned": m["unassigned_count"], "utilization %": round(m["avg_utilization_pct"], 1),
        "runtime s": round(m["runtime_s"], 3)} for n, m in evals.items()}).T
    st.dataframe(tbl, use_container_width=True)
    st.pyplot(viz.fig_comparison(evals, "Full scenario"))
    st.subheader("Reduced instance (same options for every method)")
    st.pyplot(viz.fig_reduced_comparison(red["objective_by_method"]))
    st.caption("Honest reading: CP-SAT is exact on these sizes and is expected to be best or tied. "
               "Hybrid rows show what QAOA-batched scheduling achieves; no quantum advantage is claimed.")

with tabs[5]:
    c = st.columns(2)
    with c[0]:
        st.pyplot(viz.fig_tariff(scn))
        st.pyplot(viz.fig_comparison({k: evals[k] for k in evals}, "Cost / waiting"))
    with c[1]:
        st.pyplot(viz.fig_schedule(scn, scheds[best].sessions(scn.stations), f"Timeline — {best}"))
        st.pyplot(viz.fig_utilization(scn, evals[best], f"Utilization — {best}"))
