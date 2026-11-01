"""Recovery solvers: Greedy, CP-SAT and hybrid QAOA. All read the same `RecoveryProblem`.

Constraints enforced identically by every solver (none is silently dropped):
  ports   concurrent sessions at (s,t) <= ports minus fixed sessions
  power   sum of session kW at (s,t)   <= grid limit G_{s,t} minus fixed sessions' kW
  window  arrival / departure / operating hours / full energy: by construction of the option set
The hybrid solver enforces the aggregate-power knapsack with pairwise QUBO penalties plus an exact
check on every decoded sample; see `qubo.build_qubo`.
"""
from __future__ import annotations

import time
from dataclasses import replace

from ..classical import solve_cpsat
from ..hybrid import BatchLog, HybridConfig, diverse_top_k
from ..models import Option
from ..qaoa import QAOAConfig, run_qaoa
from ..qubo import build_qubo
from .capacity import Capacity
from .problem import RecoveryProblem


def _order(p: RecoveryProblem, ids: list[int]) -> list[int]:
    """Critical first, then confirmed, ...; ties: fewest options (most constrained) first."""
    return sorted(ids, key=lambda i: (-p.priority[i], len(p.options[i]), i))


def greedy_recover(p: RecoveryProblem) -> dict[int, Option]:
    cap = Capacity(p.scn, p.fixed)
    got: dict[int, Option] = {}
    for i in _order(p, p.free_ids):
        cand = [o for o in p.options[i] if cap.fits(o)]
        if cand:
            best = min(cand, key=lambda o: (o.cost, o.start))
            got[i] = best
            cap.add(best)
    return got


def cpsat_recover(p: RecoveryProblem, time_limit: float = 20.0) -> tuple[dict[int, Option], dict]:
    cap = Capacity(p.scn, p.fixed)
    chosen, status, obj = solve_cpsat(p.options, cap.ports_free, 10.0, time_limit,
                                      penalty_by_ev=p.penalty, power_residual=cap.power_free)
    return chosen, {"status": status, "objective": obj}


def refine(p: RecoveryProblem, assigned: dict[int, Option], max_rounds: int = 10
           ) -> tuple[dict[int, Option], int]:
    """First-improvement local search on the free EVs (classical post-processing)."""
    cur, moves = dict(assigned), 0
    for _ in range(max_rounds):
        improved = False
        for i in _order(p, p.free_ids):
            cap = Capacity(p.scn, {**p.fixed, **{k: v for k, v in cur.items() if k != i}})
            cand = [o for o in p.options[i] if cap.fits(o)]
            if not cand:
                continue
            best = min(cand, key=lambda o: (o.cost, o.start))
            old = cur.get(i)
            if (old is None and best.cost < p.penalty[i] - 1e-9) or (
                    old is not None and best.cost < old.cost - 1e-9):
                cur[i] = best
                moves += 1
                improved = True
        if not improved:
            break
    return cur, moves


def _null_option(ev_id: int, penalty: float) -> Option:
    """'EV stays unserved' decision variable: zero slots, cost = unserved penalty."""
    return Option(ev_id, -1, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cost=penalty, power_kw=0.0)


def hybrid_recover(p: RecoveryProblem, qcfg: QAOAConfig, hcfg: HybridConfig | None = None
                   ) -> tuple[dict[int, Option], list[BatchLog], dict]:
    """Rolling-horizon QAOA over small batches of the free EVs, then classical refinement.

    Non-critical EVs get an explicit 'unserved' variable priced at their penalty so QAOA can
    represent contention; critical EVs get none (a batch that cannot serve them falls back to the
    classical greedy placement and is flagged, never hidden)."""
    hcfg = hcfg or HybridConfig()
    t0 = time.perf_counter()
    committed: dict[int, Option] = {}
    unserved_choice: set[int] = set()
    logs: list[BatchLog] = []
    ids = _order(p, [i for i in p.free_ids if p.options[i]])
    batches = [ids[k:k + hcfg.batch_size] for k in range(0, len(ids), hcfg.batch_size)]
    for bi, batch in enumerate(batches):
        cap = Capacity(p.scn, {**p.fixed, **committed})
        q, qres, note = None, None, ""
        for k in range(hcfg.options_per_ev, 0, -1):
            sub: dict[int, list[Option]] = {}
            for i in batch:
                fit = [o for o in p.options[i] if cap.fits(o)]
                picks = diverse_top_k(fit, k)
                keep = p.prev.get(i)
                keep_opt = next((o for o in fit if keep and (o.station_id, o.start) ==
                                 (keep.station_id, keep.start)), None)
                if keep_opt is not None and keep_opt not in picks:
                    picks = picks[:-1] + [keep_opt] if len(picks) >= k else picks + [keep_opt]
                if p.priority[i] < 3:
                    picks = picks + [_null_option(i, p.penalty[i])]
                if any(o.station_id >= 0 for o in picks):
                    sub[i] = picks
            if not sub:
                break
            q = build_qubo(sub, cap.ports_free, power_residual=cap.power_free)
            if q.n <= hcfg.max_qubits:
                break
        if q is None or q.n_decision == 0:
            logs.append(BatchLog(bi, batch, 0, 0, 0, None, False, [], "no feasible options left"))
            continue
        got: dict[int, Option] = {}
        fallback = True
        if q.n > hcfg.max_qubits:
            note = f"QUBO has {q.n} > {hcfg.max_qubits} qubits even with k=1; greedy fallback"
        else:
            try:
                qres = run_qaoa(q, qcfg)
            except Exception as exc:           # keep the app alive: classical fallback
                note = f"QAOA failed ({type(exc).__name__}); greedy fallback"
        if qres is not None and qres.best_feasible_bits is not None:
            for o in q.decode(qres.best_feasible_bits):
                if o.station_id >= 0:
                    got[o.ev_id] = o
                else:
                    unserved_choice.add(o.ev_id)
            fallback = False
        else:
            note = note or "QAOA produced no feasible sample; greedy fallback"
            c2 = Capacity(p.scn, {**p.fixed, **committed})
            for i in batch:
                cand = [o for o in p.options[i] if c2.fits(o)]
                if cand:
                    got[i] = min(cand, key=lambda o: o.cost)
                    c2.add(got[i])
        committed.update(got)
        logs.append(BatchLog(bi, batch, q.n, q.n_decision, q.n_slack, qres, fallback,
                             sorted(got), note))
    final, moves = refine(p, committed) if hcfg.refine else (dict(committed), 0)
    meta = {"batches": len(logs), "fallback_batches": sum(l.used_fallback for l in logs),
            "refinement_moves": moves, "quantum_runtime_s": time.perf_counter() - t0,
            "qaoa_runs": sum(1 for l in logs if l.qaoa is not None)}
    return final, logs, meta
