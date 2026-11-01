"""Matplotlib figures built from real optimizer outputs (each returns a Figure)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import Scenario, Session

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#0ea5e9", "#ec4899", "#84cc16"]
TIER_COLOR = {"peak": "#ef4444", "normal": "#f59e0b", "offpeak": "#10b981"}


def _slot_labels(scn: Scenario) -> list[str]:
    c = scn.config
    out = []
    for t in range(c.n_slots):
        h = (c.start_hour + t * c.slot_hours) % 24
        out.append(f"{int(h):02d}:{int(round((h % 1) * 60)):02d}")
    return out


def fig_tariff(scn: Scenario):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    T = scn.config.n_slots
    ax.bar(range(T), scn.prices, color=[TIER_COLOR[t] for t in scn.price_tiers], width=0.9)
    ax.set_xticks(range(T))
    ax.set_xticklabels(_slot_labels(scn), rotation=60, fontsize=8)
    ax.set_ylabel("INR / kWh")
    ax.set_title("Electricity tariff per slot")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TIER_COLOR.values()]
    ax.legend(handles, TIER_COLOR.keys(), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_schedule(scn: Scenario, sessions: list[Session], title: str = "Charging schedule"):
    """Gantt: one row per (station, port); bars are EV sessions."""
    rows = [(s.station_id, p) for s in scn.stations for p in range(s.ports)]
    ridx = {r: i for i, r in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.6))
    names = {e.ev_id: e.name for e in scn.evs}
    for s in sessions:
        if s.port < 0:
            continue
        y = ridx[(s.station_id, s.port)]
        ax.barh(y, s.duration, left=s.start, color=PALETTE[s.ev_id % len(PALETTE)],
                edgecolor="white", height=0.7)
        ax.text(s.start + s.duration / 2, y, names[s.ev_id].replace("EV", ""), ha="center",
                va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{scn.stations[sid].name}·P{p + 1}" for sid, p in rows], fontsize=8)
    ax.set_xticks(range(0, scn.config.n_slots + 1, 2))
    labels = _slot_labels(scn) + ["end"]
    ax.set_xticklabels([labels[i] for i in range(0, scn.config.n_slots + 1, 2)], fontsize=8)
    ax.set_xlim(0, scn.config.n_slots)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def fig_utilization(scn: Scenario, metrics: dict, title: str = "Station utilization"):
    occ = np.array(metrics["occupancy"], dtype=float)
    ports = np.array([s.ports for s in scn.stations], dtype=float)[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 2]})
    names = list(metrics["utilization_pct"])
    axes[0].bar(names, [metrics["utilization_pct"][n] for n in names], color=PALETTE[0])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("% of port-slots used")
    axes[0].set_title(title)
    im = axes[1].imshow(occ / ports * 100, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names)
    axes[1].set_xticks(range(0, occ.shape[1], 2))
    axes[1].set_xticklabels(_slot_labels(scn)[::2], fontsize=8, rotation=45)
    axes[1].set_title("Port occupancy (% of ports busy)")
    fig.colorbar(im, ax=axes[1], fraction=0.03)
    fig.tight_layout()
    return fig


def fig_qaoa_progress(history: list[dict], exact_cost: float | None = None, title: str = "QAOA optimization history"):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    it = [h["iter"] for h in history]
    ax.plot(it, [h["expected_energy"] for h in history], color=PALETTE[0], label="objective (sampled expectation)")
    ax.plot(it, [h["best_energy"] for h in history], color=PALETTE[2], label="best measured QUBO energy so far")
    if exact_cost is not None:
        ax.axhline(exact_cost, color=PALETTE[3], ls="--", label="exact QUBO ground state")
    ax.set_xlabel("objective evaluation (each = one Aer sampling run)")
    ax.set_ylabel("QUBO energy")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_distribution(counts: dict, energies: dict, feasible: dict, top: int = 12):
    keys = list(counts)[:top]
    tot = sum(counts.values())
    fig, ax = plt.subplots(figsize=(8, 3.4))
    cols = [PALETTE[2] if feasible[k] else PALETTE[3] for k in keys]
    ax.bar(range(len(keys)), [counts[k] / tot for k in keys], color=cols)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{energies[k]:.2f}" for k in keys], rotation=45, fontsize=8)
    ax.set_xlabel("QUBO energy of bitstring (green = feasible, red = violates constraints)")
    ax.set_ylabel("probability")
    ax.set_title("Final QAOA sample distribution (top outcomes)")
    fig.tight_layout()
    return fig


def fig_reduced_comparison(table: dict[str, float | None], title: str = "Quantum vs classical on the reduced instance"):
    names = [k for k, v in table.items() if v is not None]
    vals = [table[k] for k in names]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(range(len(names)), vals, color=[PALETTE[4] if "QAOA" in n else PALETTE[0] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("objective (lower = better)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_comparison(rows: dict[str, dict], title: str = "Algorithm comparison"):
    """rows: name -> metrics dict from evaluate_sessions."""
    names = list(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    comp = [("Energy", "energy_cost_inr"), ("Waiting", "waiting_cost_inr"), ("Travel", "travel_cost_inr")]
    bottom = np.zeros(len(names))
    for (lab, key), c in zip(comp, PALETTE):
        vals = np.array([rows[n][key] for n in names])
        axes[0].bar(names, vals, bottom=bottom, label=lab, color=c)
        bottom += vals
    axes[0].set_title("Cost breakdown (INR)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(names, [rows[n]["avg_wait_h"] for n in names], color=PALETTE[1])
    axes[1].set_title("Avg waiting time (h)")
    axes[2].bar(names, [rows[n]["objective"] for n in names], color=PALETTE[4])
    axes[2].set_title("Normalized objective (lower = better)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
