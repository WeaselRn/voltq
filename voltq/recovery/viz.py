"""Dark operations-dashboard figures for the disruption simulator (matplotlib, built from real schedules)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..models import Option, Scenario, grid_limit_kw
from .commitments import CommitmentType

BG, PANEL, GRID, TEXT, MUTED = "#0f1620", "#16202c", "#2a3646", "#e6edf3", "#8b98a9"
CT_COLOR = {CommitmentType.CRITICAL: "#ef4444", CommitmentType.CONFIRMED: "#22c55e",
            CommitmentType.FLEXIBLE: "#f59e0b", CommitmentType.UNCONFIRMED: "#38bdf8"}
STRAT_COLOR = {"Static (no recovery)": "#64748b", "Greedy recovery": "#f59e0b",
               "CP-SAT recovery": "#38bdf8", "Hybrid QAOA recovery": "#a78bfa",
               "CP-SAT full-network": "#2dd4bf"}


def _style(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    _style(ax)
    return fig, ax


def slot_labels(scn: Scenario) -> list[str]:
    c = scn.config
    out = []
    for t in range(c.n_slots + 1):
        h = (c.start_hour + t * c.slot_hours) % 24
        out.append(f"{int(h):02d}:{int(round((h % 1) * 60)):02d}")
    return out


def fig_gantt(scn: Scenario, assignments: dict[int, Option], ports: dict[int, int], commitments: dict,
              title: str, t_now: int | None = None, down: dict[int, int] | None = None,
              moved: set[int] | None = None, new: set[int] | None = None, ref: dict | None = None):
    """Stations x time Gantt. Colour = commitment type, thick white outline = reservation changed,
    dashed = new arrival, red hatch = station out of service, vertical line = event time."""
    from ..models import assign_ports
    sessions = assign_ports(assignments, scn.stations)
    rows = [(s.station_id, p) for s in scn.stations for p in range(s.ports)]
    ridx = {r: i for i, r in enumerate(rows)}
    fig, ax = _fig(10.5, 0.52 * len(rows) + 1.5)
    T = scn.config.n_slots
    down = down or {}
    for (sid, p), y in ridx.items():
        if sid in down:
            ax.barh(y, T - down[sid], left=down[sid], color="#7f1d1d", alpha=0.35, height=0.9,
                    hatch="///", edgecolor="#ef4444", linewidth=0)
    for s in sessions:
        if s.port < 0:
            continue
        y = ridx[(s.station_id, s.port)]
        ct = commitments[s.ev_id].ctype
        ch = moved is not None and s.ev_id in moved
        nw = new is not None and s.ev_id in new
        ax.barh(y, s.duration, left=s.start, height=0.72, color=CT_COLOR[ct], alpha=0.92,
                edgecolor="white" if ch else BG, linewidth=2.2 if ch else 1,
                linestyle="--" if nw else "-")
        ax.text(s.start + s.duration / 2, y, scn.evs[s.ev_id].name.replace("EV", ""), ha="center",
                va="center", color="#0b1118", fontsize=8, fontweight="bold")
    if t_now:
        ax.axvline(t_now, color="#e879f9", lw=1.6, ls=":")
        ax.text(t_now, -0.75, " event", color="#e879f9", fontsize=8, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{scn.stations[sid].name} · P{p + 1}" for sid, p in rows], fontsize=8)
    lab = slot_labels(scn)
    ax.set_xticks(range(0, T + 1, 2))
    ax.set_xticklabels([lab[i] for i in range(0, T + 1, 2)])
    ax.set_xlim(0, T)
    ax.set_ylim(len(rows) - 0.4, -1.0)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=GRID, alpha=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CT_COLOR.values()]
    leg = ax.legend(handles, [k.value for k in CT_COLOR], ncol=4, fontsize=7, frameon=False,
                    loc="lower right", bbox_to_anchor=(1.0, 1.0))
    for t in leg.get_texts():
        t.set_color(MUTED)
    fig.tight_layout()
    return fig


def fig_strategy_bars(rows: list[dict]):
    """Small multiples, one per headline metric; the best bar of each panel gets a white outline."""
    specs = [("recovery_runtime_s", "Recovery runtime (s)", False),
             ("total_cost_inr", "Operating cost (INR)", False),
             ("avg_wait_min", "Average wait (min)", False),
             ("reassigned_reservations", "Reassigned reservations", False),
             ("commitment_violations", "Commitment violations", False),
             ("sla_success_pct", "Departure SLA success (%)", True)]
    names = [r["strategy"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.6))
    fig.patch.set_facecolor(BG)
    for ax, (key, label, higher) in zip(axes.ravel(), specs):
        _style(ax)
        vals = np.array([r[key] for r in rows], dtype=float)
        best = vals.max() if higher else vals.min()
        for i, (n, v) in enumerate(zip(names, vals)):
            is_best = abs(v - best) < 1e-9
            ax.bar(i, v, color=STRAT_COLOR.get(n, "#94a3b8"), edgecolor="white" if is_best else "none",
                   linewidth=1.8)
            ax.text(i, v, f"{v:.2f}" if v < 100 else f"{v:.0f}", ha="center", va="bottom",
                    color=TEXT, fontsize=7)
        ax.set_xticks([])
        ax.set_title(label, fontsize=9, loc="left")
        ax.set_ylim(0, max(vals.max() * 1.18, 1e-9))
    handles = [plt.Rectangle((0, 0), 1, 1, color=STRAT_COLOR.get(n, "#94a3b8")) for n in names]
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=8,
               labelcolor=MUTED)
    fig.text(0.01, 0.965, "white outline = best value on that metric (ties shared)", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


def fig_station_power(scn: Scenario, kw: np.ndarray, station_id: int, title: str):
    """Aggregate charging power at one station vs its grid limit G_{s,t}."""
    st = scn.stations[station_id]
    T = scn.config.n_slots
    fig, ax = _fig(8.5, 3.0)
    lim = [grid_limit_kw(st, t) for t in range(T)]
    ax.bar(range(T), kw[station_id], color="#38bdf8", alpha=0.9, width=0.85, label="charging power")
    ax.step(range(T + 1), lim + [lim[-1]], where="post", color="#ef4444", lw=2, label="grid limit G(s,t)")
    ax.set_xticks(range(0, T, 2))
    ax.set_xticklabels(slot_labels(scn)[0:T:2])
    ax.set_ylabel("kW")
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(MUTED)
    ax.grid(axis="y", color=GRID, alpha=0.6)
    fig.tight_layout()
    return fig


def fig_network_utilization(scn: Scenario, occupancy: list[list[int]], title: str):
    occ = np.array(occupancy, dtype=float)
    ports = np.array([s.ports for s in scn.stations], dtype=float)[:, None]
    fig, ax = _fig(8.5, 0.5 * len(scn.stations) + 1.4)
    im = ax.imshow(occ / ports * 100, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_yticks(range(len(scn.stations)))
    ax.set_yticklabels([s.name for s in scn.stations])
    T = scn.config.n_slots
    ax.set_xticks(range(0, T, 2))
    ax.set_xticklabels(slot_labels(scn)[0:T:2])
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    return fig
