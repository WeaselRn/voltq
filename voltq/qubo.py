"""QUBO formulation of a (sub)problem and QUBO -> Ising conversion.

Binary variables
    x_o in {0,1}, one per candidate option o=(v,s,t): "EV v charges at station s from slot t".
    y_k in {0,1}, slack bits that encode capacity inequalities with residual capacity >= 2.

Energy function (minimized)
    E(x,y) = sum_o c_o x_o                                   # normalized weighted cost
           + P_a * sum_v (sum_{o in v} x_o - 1)^2            # each EV exactly one assignment
           + P_c * sum_{(o,o') conflicting, r=1} x_o x_o'    # station full: residual capacity 1
           + P_c * sum_{(s,t): r>=2} (sum_{o covers (s,t)} x_o + sum_k w_k y_k - r)^2

The time-window / energy-delivery / operating-hour constraints (3,4,5,6) are enforced
*exactly by construction*: options are only generated if the whole session fits the window and
delivers the full required energy, so no penalty is needed for them.
Constraint 7 (no overlap on a port) follows from the capacity constraint through interval-
partition port assignment; constraint 8 (no simultaneous sessions for an EV) follows from
the exactly-one constraint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .models import Option


@dataclass
class QUBO:
    n: int
    linear: np.ndarray
    quad: dict[tuple[int, int], float]
    offset: float
    options: list[Option]                       # variable i < len(options) <-> options[i]
    ev_groups: dict[int, list[int]]
    capacity: list[tuple[list[int], int, tuple[int, int]]]   # (decision var idx, residual, (s,t))
    power: list[tuple[list[int], list[float], float, tuple[int, int]]] = field(default_factory=list)
    n_slack: int = 0
    penalties: dict[str, float] = field(default_factory=dict)
    dropped: int = 0

    @property
    def n_decision(self) -> int:
        return len(self.options)

    def energy(self, bits: np.ndarray) -> float:
        b = np.asarray(bits, dtype=float)
        e = float(self.linear @ b) + self.offset
        for (i, j), v in self.quad.items():
            e += v * b[i] * b[j]
        return e

    def energies(self, X: np.ndarray) -> np.ndarray:
        """Vectorised energies for a (m, n) 0/1 matrix."""
        X = X.astype(float)
        e = X @ self.linear + self.offset
        for (i, j), v in self.quad.items():
            e += v * X[:, i] * X[:, j]
        return e

    def violations(self, bits: np.ndarray) -> dict[str, int]:
        """Independent constraint check on the decision variables (ignores penalties)."""
        b = np.asarray(bits).astype(int)
        bad_assign = sum(1 for idx in self.ev_groups.values() if int(b[idx].sum()) != 1)
        bad_cap = sum(1 for idx, r, _ in self.capacity if int(b[idx].sum()) > r)
        bad_pow = sum(1 for idx, w, lim, _ in self.power
                      if float(np.dot(w, b[idx])) > lim + 1e-9)
        return {"assignment": bad_assign, "capacity": bad_cap, "power": bad_pow}

    def is_feasible(self, bits: np.ndarray) -> bool:
        v = self.violations(bits)
        return v["assignment"] == 0 and v["capacity"] == 0 and v["power"] == 0

    def decode(self, bits: np.ndarray) -> list[Option]:
        """Selected options (decision variables only)."""
        return [self.options[i] for i in range(self.n_decision) if int(bits[i]) == 1]

    def decision_cost(self, bits: np.ndarray) -> float:
        return float(sum(o.cost for o in self.decode(bits)))


def _add_square(lin: np.ndarray, quad: dict, terms: list[tuple[int, float]], target: float, P: float) -> float:
    """Add P * (sum_i w_i z_i - target)^2 for binary z; returns the constant."""
    for a, (i, wi) in enumerate(terms):
        lin[i] += P * (wi * wi - 2 * target * wi)       # z^2 = z
        for j, wj in terms[a + 1:]:
            key = (min(i, j), max(i, j))
            quad[key] = quad.get(key, 0.0) + 2 * P * wi * wj
    return P * target * target


def slack_weights(residual: int) -> list[int]:
    """Bounded binary encoding of an integer slack in [0, residual]."""
    k = max(1, math.ceil(math.log2(residual + 1)))
    w = [2 ** i for i in range(k - 1)]
    w.append(residual - sum(w))
    return w


def build_qubo(options_by_ev: dict[int, list[Option]],
               residual: Callable[[int, int], int],
               penalty_scale: float = 1.5,
               power_residual: Callable[[int, int], float] | None = None) -> QUBO:
    """Build the QUBO for the given candidate options.

    `residual(s, t)` returns free ports at station s in slot t (ports minus already committed).
    `power_residual(s, t)` (optional) returns free grid kW. Aggregate power  sum_o p_o x_o <= G is a
    weighted knapsack row; encoding it exactly would need many slack qubits, so it is enforced as
    (a) options that never fit alone are dropped, (b) a pairwise penalty P_c x_o x_o' for every pair
    whose joint power exceeds G, and (c) an exact check on every decoded sample (`violations`) so a
    sample that breaks a >=3-way power limit is never reported feasible.
    Options that cannot fit even alone are dropped (counted in `dropped`).
    """
    opts: list[Option] = []
    dropped = 0
    for evid in sorted(options_by_ev):
        for o in options_by_ev[evid]:
            if all(residual(o.station_id, t) >= 1 for t in o.slots) and (
                    power_residual is None
                    or all(power_residual(o.station_id, t) >= o.power_kw - 1e-9 for t in o.slots)):
                opts.append(o)
            else:
                dropped += 1
    n_dec = len(opts)
    cmax = max((o.cost for o in opts), default=1.0)
    P_a = penalty_scale * (cmax + 0.5)
    P_c = P_a

    groups: dict[int, list[int]] = {}
    for i, o in enumerate(opts):
        groups.setdefault(o.ev_id, []).append(i)

    # capacity sets: (s,t) -> decision vars covering that slot
    cover: dict[tuple[int, int], list[int]] = {}
    for i, o in enumerate(opts):
        for t in o.slots:
            cover.setdefault((o.station_id, t), []).append(i)
    cap_sets: dict[tuple[frozenset, int], tuple[int, int]] = {}
    for (s, t), idx in cover.items():
        r = residual(s, t)
        if len(idx) > r:
            cap_sets.setdefault((frozenset(idx), r), (s, t))

    # slack variables only for r >= 2
    n_slack = sum(len(slack_weights(r)) for (_, r) in cap_sets if r >= 2)
    n = n_dec + n_slack
    lin = np.zeros(n)
    quad: dict[tuple[int, int], float] = {}
    offset = 0.0

    for i, o in enumerate(opts):
        lin[i] += o.cost
    for idx in groups.values():
        offset += _add_square(lin, quad, [(i, 1.0) for i in idx], 1.0, P_a)

    capacity = []
    pair_done: set[tuple[int, int]] = set()
    slack_ptr = n_dec
    for (idxset, r), (s, t) in cap_sets.items():
        idx = sorted(idxset)
        capacity.append((idx, r, (s, t)))
        if r == 1:
            for a, i in enumerate(idx):
                for j in idx[a + 1:]:
                    if opts[i].ev_id == opts[j].ev_id or (i, j) in pair_done:
                        continue          # same-EV pairs are already penalised by exactly-one
                    pair_done.add((i, j))
                    quad[(i, j)] = quad.get((i, j), 0.0) + P_c
        else:
            w = slack_weights(r)
            terms = [(i, 1.0) for i in idx] + [(slack_ptr + k, float(wk)) for k, wk in enumerate(w)]
            slack_ptr += len(w)
            offset += _add_square(lin, quad, terms, float(r), P_c)

    power_rows = []
    if power_residual is not None:
        for (s, t), idx in cover.items():
            lim = power_residual(s, t)
            w = [opts[i].power_kw for i in idx]
            if sum(w) > lim + 1e-9:
                power_rows.append((list(idx), w, float(lim), (s, t)))
                for a, i in enumerate(idx):
                    for j in idx[a + 1:]:
                        if opts[i].ev_id == opts[j].ev_id:
                            continue
                        if opts[i].power_kw + opts[j].power_kw > lim + 1e-9:
                            key = (min(i, j), max(i, j))
                            if key not in pair_done:
                                pair_done.add(key)
                                quad[key] = quad.get(key, 0.0) + P_c

    return QUBO(n=n, linear=lin, quad=quad, offset=offset, options=opts, ev_groups=groups,
                capacity=capacity, power=power_rows, n_slack=n_slack,
                penalties={"assignment": P_a, "capacity": P_c}, dropped=dropped)


def to_ising(q: QUBO) -> tuple[np.ndarray, dict[tuple[int, int], float], float]:
    """x = (1 - z)/2.  E = const + sum h_i z_i + sum J_ij z_i z_j."""
    h = -q.linear / 2.0
    const = q.offset + q.linear.sum() / 2.0
    J: dict[tuple[int, int], float] = {}
    for (i, j), b in q.quad.items():
        h[i] -= b / 4.0
        h[j] -= b / 4.0
        J[(i, j)] = b / 4.0
        const += b / 4.0
    return h, J, const


def ising_energy(h: np.ndarray, J: dict, const: float, bits: np.ndarray) -> float:
    z = 1 - 2 * np.asarray(bits, dtype=float)
    return const + float(h @ z) + sum(v * z[i] * z[j] for (i, j), v in J.items())


def exact_solve(q: QUBO, max_vars: int = 22) -> dict:
    """Brute-force reference: global minimum of E and best *feasible* decision assignment.

    Slack bits are enumerated too, so the minimum is of the full QUBO.
    """
    if q.n > max_vars:
        raise ValueError(f"{q.n} variables > {max_vars}: brute force skipped")
    best_e, best_b = np.inf, None
    best_fe, best_fb = np.inf, None
    chunk = 1 << min(q.n, 16)
    total = 1 << q.n
    shifts = np.arange(q.n)
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total))
        X = ((idx[:, None] >> shifts) & 1)
        E = q.energies(X)
        k = int(np.argmin(E))
        if E[k] < best_e:
            best_e, best_b = float(E[k]), X[k].copy()
        # feasibility mask on decision vars
        feas = np.ones(len(idx), dtype=bool)
        for g in q.ev_groups.values():
            feas &= X[:, g].sum(axis=1) == 1
        for cidx, r, _ in q.capacity:
            feas &= X[:, cidx].sum(axis=1) <= r
        for pidx, w, lim, _ in q.power:
            feas &= (X[:, pidx] * np.array(w)).sum(axis=1) <= lim + 1e-9
        if feas.any():
            Ef = np.where(feas, E, np.inf)
            k = int(np.argmin(Ef))
            if Ef[k] < best_fe:
                best_fe, best_fb = float(Ef[k]), X[k].copy()
    return {"min_energy": best_e, "min_bits": best_b,
            "min_feasible_energy": best_fe if best_fb is not None else None,
            "min_feasible_bits": best_fb}
