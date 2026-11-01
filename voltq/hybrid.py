"""Hybrid quantum-classical scheduling (rolling horizon).

1. classical: feasible options per EV (preprocess.build_options)
2. classical: sort EVs by arrival/urgency, cut into batches of `batch_size`  (rolling horizon)
3. per batch: residual capacity from already committed sessions, keep the k cheapest options per
   EV (diverse across stations), build a QUBO
4. QUANTUM (Aer): QAOA samples the batch QUBO; best *feasible* decoded sample is committed
   - if QAOA finds no feasible sample, the batch falls back to greedy (flagged in the log)
5. classical refinement: local search (re-insert unassigned EVs, move EVs to cheaper options)
6. classical validation (evaluate.py)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .classical import committed_residual, fits
from .models import Option, Scenario, Schedule
from .preprocess import Preprocessed
from .qaoa import QAOAConfig, QAOAResult, run_qaoa
from .qubo import build_qubo


@dataclass
class HybridConfig:
    batch_size: int = 3
    options_per_ev: int = 3
    max_qubits: int = 18
    refine: bool = True


@dataclass
class BatchLog:
    batch: int
    ev_ids: list[int]
    n_qubits: int
    n_decision: int
    n_slack: int
    qaoa: QAOAResult | None
    used_fallback: bool
    committed: list[int]
    note: str = ""


def diverse_top_k(opts: list[Option], k: int) -> list[Option]:
    """Cheapest option per station first (diversity), then fill with cheapest remaining."""
    ordered = sorted(opts, key=lambda o: (o.cost, o.start))
    picked: list[Option] = []
    seen: set[int] = set()
    for o in ordered:
        if o.station_id not in seen and len(picked) < k:
            picked.append(o)
            seen.add(o.station_id)
    for o in ordered:
        if len(picked) >= k:
            break
        if o not in picked:
            picked.append(o)
    return picked


def make_batches(scn: Scenario, pre: Preprocessed, size: int) -> list[list[int]]:
    order = sorted((e for e in scn.evs if pre.options[e.ev_id]),
                   key=lambda e: (e.arrival_slot, e.departure_slot - e.arrival_slot, e.ev_id))
    ids = [e.ev_id for e in order]
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def refine_local_search(scn: Scenario, pre: Preprocessed, assigned: dict[int, Option],
                        max_rounds: int = 20) -> tuple[dict[int, Option], int]:
    """First-improvement local search; returns (assignments, number of improving moves)."""
    pen = scn.config.unassigned_penalty
    moves = 0
    cur = dict(assigned)
    for _ in range(max_rounds):
        improved = False
        for ev in scn.evs:
            rest = {k: v for k, v in cur.items() if k != ev.ev_id}
            res = committed_residual(scn, rest)
            cand = [o for o in pre.options[ev.ev_id] if fits(o, res)]
            if not cand:
                continue
            best = min(cand, key=lambda o: (o.cost, o.start))
            old = cur.get(ev.ev_id)
            if old is None and best.cost < pen:
                cur[ev.ev_id] = best
                moves += 1
                improved = True
            elif old is not None and best.cost < old.cost - 1e-9:
                cur[ev.ev_id] = best
                moves += 1
                improved = True
        if not improved:
            break
    return cur, moves


def hybrid_schedule(scn: Scenario, pre: Preprocessed, qcfg: QAOAConfig,
                    hcfg: HybridConfig | None = None) -> tuple[Schedule, Schedule, list[BatchLog]]:
    """Returns (schedule before refinement, final schedule, per-batch logs)."""
    hcfg = hcfg or HybridConfig()
    t0 = time.perf_counter()
    committed: dict[int, Option] = {}
    logs: list[BatchLog] = []
    for bi, ids in enumerate(make_batches(scn, pre, hcfg.batch_size)):
        res = committed_residual(scn, committed)
        # shrink k until the batch fits the qubit budget
        q = None
        for k in range(hcfg.options_per_ev, 0, -1):
            sub = {i: diverse_top_k([o for o in pre.options[i] if fits(o, res)], k) for i in ids}
            sub = {i: v for i, v in sub.items() if v}
            if not sub:
                break
            q = build_qubo(sub, res)
            if q.n <= hcfg.max_qubits:
                break
        if q is None or q.n_decision == 0:
            logs.append(BatchLog(bi, ids, 0, 0, 0, None, False, [], "no feasible options left"))
            continue
        if q.n > hcfg.max_qubits:
            q_note = f"QUBO has {q.n} > {hcfg.max_qubits} qubits even with k=1; greedy fallback"
            qres = None
        else:
            q_note = ""
            qres = run_qaoa(q, qcfg)
        got: dict[int, Option] = {}
        fallback = True
        if qres is not None and qres.best_feasible_bits is not None:
            got = {o.ev_id: o for o in q.decode(qres.best_feasible_bits)}
            fallback = False
        else:
            q_note = q_note or "QAOA produced no feasible sample; greedy fallback"
            for ev_id in sorted(ids, key=lambda i: scn.evs[i].departure_slot - scn.evs[i].arrival_slot):
                r2 = committed_residual(scn, {**committed, **got})
                cand = [o for o in pre.options[ev_id] if fits(o, r2)]
                if cand:
                    got[ev_id] = min(cand, key=lambda o: o.cost)
        committed.update(got)
        logs.append(BatchLog(bi, ids, q.n, q.n_decision, q.n_slack, qres, fallback,
                             sorted(got), q_note))

    t_q = time.perf_counter() - t0
    unassigned = [e.ev_id for e in scn.evs if e.ev_id not in committed]
    raw = Schedule("Hybrid (QAOA batches)", dict(committed), unassigned, t_q,
                   {"batches": len(logs), "fallback_batches": sum(l.used_fallback for l in logs)})
    final_assign, moves = (refine_local_search(scn, pre, committed) if hcfg.refine
                           else (dict(committed), 0))
    final = Schedule("Hybrid (QAOA + refinement)", final_assign,
                     [e.ev_id for e in scn.evs if e.ev_id not in final_assign],
                     time.perf_counter() - t0,
                     {**raw.meta, "refinement_moves": moves})
    return raw, final, logs
