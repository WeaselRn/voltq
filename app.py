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
st.caption("Smart scheduling of EV charging. The quantum part (QAOA) is simulated on your own computer "
           "with Qiskit Aer. No real quantum hardware or cloud is used.")

with st.sidebar:
    st.header("Scenario")
    n_evs = st.slider("Number of EVs", 4, 15, 10)
    n_st = st.slider("Number of stations", 2, 5, 4)
    pmin, pmax = st.slider("Ports per station (min, max)", 1, 3, (2, 2))
    n_slots = st.slider("Time slots (30 min each)", 8, 16, 12)
    seed = st.number_input("Random seed", 0, 10_000, 42)
    with st.expander("Electricity price (₹ per kWh)"):
        peak = st.number_input("Peak", 1.0, 50.0, 14.0, 0.5)
        normal = st.number_input("Normal", 1.0, 50.0, 9.0, 0.5)
        off = st.number_input("Off-peak", 1.0, 50.0, 5.0, 0.5)
        start_hour = st.slider("Window starts at (hour of day)", 0.0, 23.5, 19.0, 0.5)
    with st.expander("What matters more? (weights)"):
        alpha = st.slider("Electricity cost", 0.0, 3.0, 1.0, 0.1)
        beta = st.slider("Waiting time", 0.0, 3.0, 1.0, 0.1)
        gamma = st.slider("Travel distance", 0.0, 3.0, 1.0, 0.1)
    with st.expander("Quantum (QAOA) settings"):
        depth = st.slider("Circuit layers (p)", 1, 3, 1)
        shots = st.select_slider("Samples per run (shots)", [128, 256, 512, 1024], 512)
        optimizer = st.selectbox("Tuning method", OPTIMIZERS)
        maxiter = st.slider("Tuning steps", 10, 120, 60, 10)
        cvar = st.slider("Focus on best samples (1 = off)", 0.1, 1.0, 1.0, 0.1)
        batch = st.slider("EVs per quantum batch", 2, 4, 3)
        sweep = st.checkbox("Also compare 1, 2 and 3 layers (slower)", False)
    run = st.button("Run optimization", type="primary", use_container_width=True)


@st.cache_resource(show_spinner="Optimizing (simulating quantum circuits)…")
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
EXACT = "Exact QUBO optimum (brute force)"
rc = red["random_control"]
s = qaoa_summary(qr)

tabs = st.tabs(["Overview", "Schedule", "Stations", "Quantum", "Comparison"])

with tabs[0]:
    m = evals[best]
    st.markdown(f"**In short:** {len(scn.evs)} EVs need charging at {len(scn.stations)} stations. "
                f"The cheapest valid plan (**{best}**) charges {m['completion_pct']:.0f}% of them for "
                f"**₹{m['total_cost_inr']:,.0f}** in total, with an average wait of "
                f"**{m['avg_wait_h'] * 60:.0f} min**.")
    c = st.columns(4)
    c[0].metric("Total cost", f"₹{m['total_cost_inr']:,.0f}")
    c[1].metric("Average wait", f"{m['avg_wait_h'] * 60:.0f} min")
    c[2].metric("EVs fully charged", f"{m['completion_pct']:.0f}%")
    c[3].metric("Chargers busy", f"{m['avg_utilization_pct']:.0f}%")
    unserved = [n for n, mm in evals.items() if mm["unassigned_count"]]
    if pre.infeasible_evs:
        st.warning(f"These EVs cannot be charged in time at any station: "
                   f"{', '.join(scn.evs[i].name for i in pre.infeasible_evs)}.")
    if unserved:
        st.warning("Some plans leave EVs uncharged (never hidden): "
                   + ", ".join(f"{n} ({evals[n]['unassigned_count']})" for n in unserved))
    st.subheader("Electricity price through the evening")
    st.caption("Red = expensive peak hours, green = cheap night hours. Good plans shift charging to green slots.")
    st.pyplot(viz.fig_tariff(scn))
    with st.expander("How is the quantum part used?"):
        st.write("A quantum algorithm (QAOA) picks charging slots for small groups of EVs. "
                 "A normal computer handles the rest and double-checks every plan. "
                 "A plan is only called valid if an independent checker finds no broken rule.")

with tabs[1]:
    pick = st.selectbox("Plan to view", list(scheds), index=list(scheds).index(best))
    st.markdown("**In short:** each row shows where and when one EV charges. Bars below show the same plan as a timeline.")
    if evals[pick]["violations"]:
        st.error("Checker found problems:\n" + "\n".join(evals[pick]["violations"]))
    else:
        st.success("Checker: no rules broken for any charged EV.")
    tbl = schedule_table(scn, scheds[pick])[["ev", "status", "station", "port", "start_h", "required_kwh",
                                             "energy_cost_inr", "wait_h", "distance_km"]]
    tbl.columns = ["EV", "Status", "Station", "Port", "Starts at (hour)", "Energy needed (kWh)",
                   "Electricity cost (₹)", "Wait (h)", "Drive (km)"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.pyplot(viz.fig_schedule(scn, scheds[pick].sessions(scn.stations), f"Charging timeline: {pick}"))

with tabs[2]:
    pick2 = st.selectbox("Plan to view", list(evals), index=list(evals).index(best), key="util")
    mu = evals[pick2]
    occ = np.array(mu["occupancy"])
    busiest = max(mu["utilization_pct"], key=mu["utilization_pct"].get)
    quietest = min(mu["utilization_pct"], key=mu["utilization_pct"].get)
    st.markdown(f"**In short:** on average {mu['avg_utilization_pct']:.0f}% of charger time is used. "
                f"{busiest} is busiest ({mu['utilization_pct'][busiest]:.0f}%), "
                f"{quietest} is quietest ({mu['utilization_pct'][quietest]:.0f}%).")
    rows = [{"Station": s_.name, "Ports": s_.ports, "Speed (kW)": s_.power_kw,
             "Used (%)": round(mu["utilization_pct"][s_.name]),
             "Most cars at once": int(occ[s_.station_id].max())} for s_ in scn.stations]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.pyplot(viz.fig_utilization(scn, mu, f"Station usage: {pick2}"))

with tabs[3]:
    opt = red["objective_by_method"][EXACT]
    got = s["best_feasible_cost"]
    same = red["qaoa_matches_exact"]
    st.markdown(
        f"**In short:** we gave the quantum circuit a small problem ({len(red['ev_ids'])} EVs, "
        f"{s['qubits']} qubits). It "
        f"{'found the best possible answer' if same else 'did not find the best answer'}"
        f"{'' if got is None else f' (score {got:.3f} vs best {opt:.3f}; lower is better)'}. "
        f"However, picking random answers with the same effort did too, so **this size shows no quantum advantage**.")
    c = st.columns(4)
    c[0].metric("Qubits used", s["qubits"])
    c[1].metric("Circuit steps (depth)", s["circuit_depth"])
    c[2].metric("Tuning rounds", s["evaluations"])
    c[3].metric("Simulation time", f"{s['sim_time_s']:.1f} s")
    st.subheader("How often does each method land on the best answer?")
    st.caption("Per sample drawn. Higher means the method finds the best plan more reliably.")
    cmp = pd.DataFrame({"Quantum (QAOA)": [100 * s["prob_best_feasible_final"]],
                        "Random guessing": [100 * rc.get("prob_best", 0.0)]},
                       index=["Chance a sample is the best plan (%)"]).T
    st.bar_chart(cmp)
    st.subheader("Learning curve")
    st.caption("The blue line should fall as the circuit is tuned. The dashed red line is the true best.")
    ex = red["exact"]["min_energy"] if red["exact"] else None
    st.pyplot(viz.fig_qaoa_progress(qr.history, ex))
    with st.expander("Technical details"):
        st.write(f"Layers p={s['depth']}, shots={s['shots']}, tuner={s['optimizer']}, seed={s['seed']}. "
                 f"{red['n_decision']} decision qubits + {red['n_slack']} helper qubits. Gates: {s['gate_counts']}.")
        bi, counts = red["bit_info"], qr.final_counts
        st.pyplot(viz.fig_distribution(counts, {k: bi[k][0] for k in counts}, {k: bi[k][1] for k in counts}))
        st.write(f"Valid samples: quantum {100 * s['prob_feasible_final']:.1f}%, "
                 f"random {100 * rc['prob_feasible']:.1f}%.")
        if res["depth_sweep"]:
            st.dataframe(pd.DataFrame(res["depth_sweep"]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame([{
            "batch": l.batch, "EVs": ", ".join(scn.evs[i].name for i in l.ev_ids), "qubits": l.n_qubits,
            "tuning rounds": l.qaoa.n_evaluations if l.qaoa else None,
            "used backup greedy": l.used_fallback} for l in res["hybrid_logs"]]),
            hide_index=True, use_container_width=True)

with tabs[4]:
    cl = {n: m for n, m in evals.items()}
    cheapest = min(cl, key=lambda n: cl[n]["total_cost_inr"] if cl[n]["unassigned_count"] == 0 else 1e18)
    st.markdown(f"**In short:** the exact classical solver (OR-Tools) is hard to beat at this size. "
                f"Lowest total cost among plans that charge everyone: **{cheapest}** "
                f"(₹{cl[cheapest]['total_cost_inr']:,.0f}). "
                "The hybrid plan uses the quantum circuit for each small batch; we do not claim it is better.")
    tbl = pd.DataFrame({n: {
        "Valid?": "yes" if m["feasible"] else "no",
        "Total cost (₹)": round(m["total_cost_inr"]),
        "Electricity (₹)": round(m["energy_cost_inr"]),
        "Waiting (₹)": round(m["waiting_cost_inr"]),
        "Travel (₹)": round(m["travel_cost_inr"]),
        "Avg wait (min)": round(m["avg_wait_h"] * 60),
        "EVs left uncharged": m["unassigned_count"],
        "Time to compute (s)": round(m["runtime_s"], 2)} for n, m in cl.items()}).T
    st.dataframe(tbl, use_container_width=True)
    st.pyplot(viz.fig_comparison(cl, "All plans side by side"))
    with st.expander("Quantum vs classical on the small test problem"):
        st.write("All methods get the same options. Equal bars mean they all found the best plan.")
        st.pyplot(viz.fig_reduced_comparison(red["objective_by_method"]))
