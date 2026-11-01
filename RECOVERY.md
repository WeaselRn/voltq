# VoltQ Phase 2 — commitment-aware dynamic recovery

> **Recover the charging network, not just the mathematical solution.**
> A static optimizer assumes nothing changes. VoltQ keeps customer reservations stable while the network is disrupted.

Everything is simulated and offline: Qiskit Aer for circuits, seeded synthetic data, simulated telemetry.
No IBM hardware, no live charger, no external API. **No quantum advantage is claimed** (CP-SAT matches or beats the hybrid solver in the results below).

## Run it

```bash
streamlit run app.py                   # dashboard: Disruption simulator tab is the demo centre
python -m pytest -q                    # 51 tests (17 original + 34 recovery)
python run_recovery_benchmark.py 6 12  # multi-seed benchmark -> outputs/recovery_benchmark.csv
```

## Architecture

```
voltq/recovery/commitments.py        commitment types, recovery objective weights, commitment mix
voltq/recovery/events.py             common Event interface + 6 disruption types (registry, @register)
voltq/recovery/state.py              NetworkState: schedule, commitments, history, frozen sessions
voltq/recovery/detection.py          affected-EV detection (direct / invalidated / conflicts / unserved)
voltq/recovery/problem.py            RecoveryProblem: identical input for every solver + network_objective
voltq/recovery/solvers.py            Greedy, CP-SAT, hybrid QAOA recovery (+ local-search refinement)
voltq/recovery/recovery_optimizer.py scope selection, staged expansion, strategy dispatch
voltq/recovery/recovery_metrics.py   all metrics, computed from the FINAL schedule
voltq/recovery/viz.py                dark Gantt / comparison figures
voltq/simulation/scenarios.py        7 seeded scenarios        .../network_simulator.py  trigger/recover/commit
voltq/simulation/benchmark.py        multi-seed benchmark
voltq/realtime/ocpp_like.py          OCPP-style message -> Event adapter (simulated)
```

Existing modules were extended, not rewritten: `models` (grid power, option kW), `preprocess` (min start, fixed refs),
`classical.solve_cpsat` (per-EV penalty, power rows), `qubo` (power rows), `evaluate` (grid power check).
All original tests still pass.

## Workflow

```
initial schedule (Greedy | CP-SAT | hybrid QAOA) -> commitments assigned
 -> event(s) applied to a copy of the state
 -> affected-EV detection (validator re-checks every reservation in the NEW network)
 -> identical RecoveryProblem for: Static | Greedy | CP-SAT | Hybrid QAOA | CP-SAT full-network
 -> independent validation + metrics -> before/after comparison -> optionally adopt as the new network
```

Sessions that already started (`start < t_now`) are **frozen**. An event can interrupt a running session
(outage, power cap): the EV keeps the energy it already received, stays at that station's location and re-enters the pool.

## Recovery objective

```
C_recovery = C_operational                      energy + waiting + travel (original normalised objective)
           + lam_reassign       * [moved]       any reservation that changes station or start
           + lam_commit[type]   * [moved]       confirmed 2.0, critical 3.0, flexible / unconfirmed 0
           + P_deadline[type]   * [unserved]    critical 1000, confirmed 15, flexible / unconfirmed 10
           + lam_fair * disadvantage * wait     optional fairness (0 = off)
```

All terms are **linear** in the option variables `x[v,s,t]`, so they enter every solver as option cost:

| Solver | How the terms appear |
|---|---|
| QUBO / QAOA | diagonal `c_o`; non-critical EVs also get a "stay unserved" variable priced at `P_deadline` |
| CP-SAT | objective coefficients; `u_v` (unserved) priced per EV |
| Greedy | option sort key; EVs ordered critical first |

* **Critical = hard where feasible.** `P_deadline = 1000` dominates every other term, so a critical EV is only dropped when no capacity exists; this is reported, never hidden. (Not an absolute constraint, so impossible cases stay solvable.)
* **Protection is configurable:** `soft` (default; confirmed EVs move only as a last expansion stage, penalised), `strict` (only to save a critical EV), `off` (no reassignment penalty).
* Penalties were not set arbitrarily large: they are the same order as the normalised cost (≈1–10) except the critical deadline; the benchmark below shows the stability/service trade-off they produce.

## Grid-aware charging

Per station and slot: `sum_v P_{v,s,t} <= G_{s,t}` (`Station.grid_kw`, `grid_from`), plus the port limit. Sessions keep a fixed power
`min(EV max, station per-port)` (no variable-power charging). Enforced in:

* **Greedy / CP-SAT / validator**: exactly (CP-SAT linear row per cell; validator re-derives kW from EV and station tables).
* **QUBO**: a weighted knapsack row would need many slack qubits, so it is enforced as (a) options that never fit alone are dropped, (b) a pairwise penalty `P_c x_o x_o'` for pairs whose joint kW exceeds `G`, (c) an **exact check on every decoded sample**: a ≥3-way overload is never reported feasible. Documented, not silently omitted.

## Events

| Event | Direct effect | Affected-EV detection |
|---|---|---|
| Station outage | station closes at the slot; running sessions interrupted | EVs on that station (future + interrupted) |
| New arrival | EV appended, commitment *unconfirmed* | the new EV, then expansion if it cannot be placed |
| Departure change | EV window changes | that EV (reservation re-validated) |
| Power reduction | station cap × fraction from the slot; excess running sessions paused | sessions in over-limit cells (validator) |
| Tariff change | prices × factor in a slot range | EVs whose window overlaps the range (may shift to profit) |
| Session overrun | running session keeps its port longer | later sessions that now collide |

New event types: subclass `Event`, implement `apply`, decorate with `@register`. Detection works on the resulting state, so nothing else changes.

**Local vs full recovery.** Stage 0: affected EVs only. While some EV stays unserved the set grows: flexible/unconfirmed EVs on
usable stations → all flexible/unconfirmed → confirmed EVs (always allowed if a critical EV is unserved). `CP-SAT full-network` frees every
not-yet-started EV as a comparison.

## Metrics (all from the resulting schedule)

Recovery runtime · operating cost · avg / max wait · reassigned reservations (moved + dropped) · commitment violations
(split into **forced by the disruption** vs **avoidable**) · departure-SLA success (all and critical) · unassigned EVs · unserved kWh ·
utilization · grid power violations · Jain fairness of `1/(1+wait)` · recovery objective by term. Hard violations come from the independent validator
and are separate from soft penalties. A schedule is never `valid` if any hard rule is broken.

## Results (12 EVs, 4 stations, 6 seeds × 7 scenarios, `run_recovery_benchmark.py`)

205 recovered schedules per table row set; **100% passed the independent validator**. Means:

| Strategy | Objective ↓ | SLA % | Critical SLA % | Unassigned EVs | Reassigned | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Static (no recovery) | 309.8 | 77.1 | 87.4 | 2.88 | 1.63 | 0.01 s |
| Greedy recovery | 27.9 | 92.4 | 100 | 0.95 | 1.85 | <0.01 s |
| CP-SAT recovery | **26.8** | 93.3 | 100 | 0.83 | 1.80 | 0.03 s |
| Hybrid QAOA recovery | 27.0 | 93.1 | 100 | 0.85 | 1.80 | 1.0 s |
| CP-SAT full-network | **26.8** | 93.3 | 100 | 0.83 | 1.80 | 0.05 s |

Honest reading: recovery matters (objective 310 → ~27). CP-SAT is best or tied everywhere; the hybrid solver is within ~1% of it and ahead of Greedy
on contested scenarios (station failure, combined) but never beats CP-SAT. Its 1 s is **Aer simulation time**, not hardware time.

## Quantum part in recovery

Free EVs are cut into batches of ≤3 (critical first). Per batch: residual ports/kW from fixed + already-committed sessions, k cheapest
diverse options per EV (+ the "keep" option, + "unserved" for non-critical EVs), QUBO ≤ 18 qubits, QAOA on Aer, best *feasible* decoded sample committed,
classical greedy fallback (flagged) if no feasible sample or if QAOA raises. Then local-search refinement. The Quantum engine tab shows
qubits, decision/slack variables, depth, gates, shots, evaluations, sim time, valid-sample rate and best energy per batch. Classical steps (preprocessing, batching,
decoding, refinement, validation) are not presented as quantum.

## Predefined scenarios (seeded, offline)

1 Station failure · 2 Unexpected arrivals (2 EVs) · 3 Grid power reduction (−40%) · 4 Departure change · 5 Combined (outage + arrival + departure) ·
6 Tariff change · 7 Session overrun. Targets are chosen by rule (busiest station, tightest protected EV), so they stay meaningful for any seed.

## Assumptions and limitations

* Fixed-power, contiguous sessions; interrupted sessions resume as a new session with updated SoC. No taper curve, no variable-power charging.
* An event occurs at one slot; multi-slot event timelines are chained by adopting a recovery and triggering the next event.
* Departure change cannot target an EV whose session already runs past the new departure (rejected).
* Port numbers are re-derived by interval partitioning, so a running session may show on a different port after recovery.
* QAOA sees ≤3 EVs at a time; results depend on COBYLA/shots. Quantum advantage is **not** claimed or measured.
* Commitment mix (critical / confirmed / flexible) is synthetic: tightest 20% windows are critical, the rest 30% flexible.

## Path to real integration (not implemented)

`voltq/realtime/ocpp_like.py` shows the seam: OCPP `StatusNotification` / `ChangeAvailability` → outage, `SetChargingProfile` → power cap, a tariff feed
(OCPI-style) → tariff change. Routing services would supply travel times; grid telemetry the power caps. Only simulated messages are used today.

## 5-minute demo script

1. **Network overview** — healthy network, validator PASS.
2. **Disruption simulator** → *Station failure* → Trigger. Show affected EVs, the interrupted sessions, untouched reservations.
3. *Run recovery* — before/after Gantt, white outlines = changed reservations, validator PASS, forced vs avoidable violations.
4. **Recovery comparison** — Static fails SLAs; Greedy / CP-SAT / Hybrid compared on identical inputs; run the multi-seed benchmark.
5. **Quantum engine** — QUBO, Ising, circuit, distribution, QAOA vs random vs exact; recovery batches; say plainly: no advantage claimed.
6. Chain a second event (*Adopt schedule* → *Grid power reduction*) to show the platform handling a sequence.
