# VoltQ — quantum-assisted EV charging optimization

Hybrid quantum-classical scheduling of EV charging. **All quantum circuits run locally on Qiskit Aer.**
No IBM hardware, IBM Quantum cloud, Runtime or API keys are used (`qiskit-ibm-runtime` is installed but never imported).

> **Phase 2:** commitment-aware dynamic recovery, grid-aware charging, disruption simulator and recovery benchmarks are documented in [RECOVERY.md](RECOVERY.md).

## Problem

EV demand fluctuates. Poor scheduling causes long waits, uneven station use, high energy cost and
needless driving. VoltQ decides **which station** and **which time slot** each EV charges in, minimising

```
C = alpha * C_energy + beta * C_waiting + gamma * C_travel   (+ unassigned_penalty per unserved EV)
```

## Architecture

```
voltq/data.py        seeded synthetic EVs, stations, tariff            -> data/*.csv, scenario.json
voltq/preprocess.py  energy, duration, distance, feasible options (v,s,t), normalization
voltq/qubo.py        QUBO builder, QUBO->Ising, exact brute-force reference
voltq/qaoa.py        QAOA circuit + optimizer loop on AerSimulator
voltq/classical.py   Greedy + OR-Tools CP-SAT (same objective and constraints)
voltq/hybrid.py      rolling-horizon batches -> QAOA per batch -> local-search refinement
voltq/evaluate.py    independent validator + metrics (re-derives everything from raw tables)
voltq/study.py       full study (4 schedulers, QAOA depth sweep, random-sampling control), outputs
voltq/viz.py         matplotlib figures          app.py  Streamlit dashboard
```

## Mathematical formulation

Slot length 30 min, horizon T=12. EV v: battery `B_v`, SoC `a_v`, target `b_v`, arrival `r_v`, departure `d_v`, max power `P_v`.
Station s: ports `K_s`, power `Q_s`, hours `[open_s, close_s)`, price factor `f_s`.

* Grid energy `E_v = B_v (b_v - a_v) / 100 / eta` (eta = 0.92).
* Power `p_vs = min(P_v, Q_s)`, duration `D_vs = ceil(E_v / (p_vs * 0.5h))` slots (one contiguous session).
* Earliest start at s: `r_v + ceil(dist_vs / speed / 0.5h)`. Options exist only for starts `t` with
  `t >= earliest`, `t >= open_s`, `t + D_vs <= min(d_v, close_s, T)`.

**Binary variables** `x[v,s,t] = 1` iff EV v charges at station s from slot t. Only feasible triples
become variables (e.g. 480 naive → 164 for the default scenario).

**Costs** (INR): `C_energy = sum_t price_t * f_s * kWh_t`, `C_waiting = wait_hours * 250 INR/h`,
`C_travel = dist_km * 8 INR/km`. Each component is divided by its mean over all feasible options
(dimensionless, ~1), then weighted: `c_o = alpha*E/E_ref + beta*W/W_ref + gamma*T/T_ref`.
Unserved EV costs `unassigned_penalty` (10). Weights and rates are configurable.

**QUBO** (minimised):

```
E(x) = sum_o c_o x_o
     + P_a * sum_v ( sum_{o in v} x_o - 1 )^2                       exactly one assignment per EV
     + P_c * sum_{conflicting o,o', residual capacity 1} x_o x_o'     station full
     + P_c * sum_{(s,t): residual r>=2} ( sum_{o covers (s,t)} x_o + sum_k w_k y_k - r )^2
```

`y_k` are slack bits (bounded binary encoding of `0..r`). `P_a = P_c = 1.5 (max c + 0.5)`.
Windows, operating hours and full energy delivery hold **by construction** of the option set.
Port non-overlap follows from capacity via interval-partition port assignment; an EV cannot have
simultaneous sessions because of the exactly-one term. All of this is re-checked by the independent
validator, never assumed.

**Ising / QAOA.** `x = (1 - z)/2` gives `H_C = const + sum h_i Z_i + sum J_ij Z_i Z_j`
(verified equal to the QUBO in tests). Circuit: `H` on all qubits, then p layers of
`RZ(2 gamma h_i)`, `RZZ(2 gamma J_ij)` (coefficients scaled by max |coef|) and mixer `RX(2 beta)`.
Aer samples bitstrings; the sample-mean (or CVaR) of QUBO energies is returned to a scipy optimizer
(COBYLA / Nelder-Mead / Powell). Every evaluation is logged and plotted.

## Hybrid workflow (what is quantum, what is not)

| # | Step | Where |
|---|------|-------|
| 1 | Generate data, compute feasible options, normalise costs | classical |
| 2 | Sort EVs by arrival/urgency, cut into batches of 3 (rolling horizon) | classical |
| 3 | Per batch: residual capacity from committed sessions, k=3 diverse cheapest options per EV, build QUBO (≤18 qubits) | classical |
| 4 | **QAOA on Aer** samples the batch QUBO; best feasible decoded sample is committed (greedy fallback if none) | **quantum simulation + classical optimizer** |
| 5 | Local-search refinement (insert unserved EVs, move EVs to cheaper options) | classical |
| 6 | Independent validation and metrics | classical |

The 10–15 EV problem is **not** solved by one QAOA circuit. Only small batch QUBOs and one reduced
"demonstrator" instance (3 EVs) are sampled on Aer.

## Install and run

Environment is the existing `qiskit-env` (Python 3.12, qiskit 2.6.1, qiskit-aer 0.18.2, qiskit-optimization 0.7.0,
OR-Tools 9.15). Streamlit was added for the dashboard.

```
qiskit-env\Scripts\activate
python run_phase1.py                 # quick demo: baselines + reduced-instance QAOA
python run_demo.py                   # full: hybrid, depth sweep, all comparisons, outputs/
python run_demo.py --evs 15 --stations 5 --ports-min 1 --ports-max 2 --seed 7 --depth 2
streamlit run app.py                 # dashboard (live scenario + QAOA configuration)
python -m pytest -q                  # 17 tests
```

Outputs: `data/` (evs, stations, prices, distance_matrix CSV, scenario.json), `outputs/` (schedule CSVs,
`optimized_schedule.csv`, `comparison.json`, `figures/`: assignment, utilization, tariff, qaoa_progress,
quantum_vs_classical, comparison).

## Sample results (seed 42, 10 EVs, 4 stations × 2 ports, 12 slots; produced by `run_demo.py`)

| Method | Feasible (validated) | Cost INR | Objective | Avg wait h | Unassigned | Runtime s |
|---|---|---|---|---|---|---|
| Greedy | yes | 4904.29 | 18.530 | 0.00 | 0 | 0.00 |
| OR-Tools CP-SAT | yes (OPTIMAL) | 4904.29 | 18.530 | 0.00 | 0 | 0.03 |
| Hybrid (QAOA batches) | yes | 5068.88 | 18.592 | 0.05 | 0 | 2.1 |
| Hybrid (+ refinement) | yes | 5068.88 | 18.592 | 0.05 | 0 | 2.1 |

Congested scenario (`--evs 15 --stations 5 --ports-min 1 --ports-max 2 --seed 7`): Greedy objective 36.46,
CP-SAT 35.13 (optimal), Hybrid 36.00; every method leaves 1 EV unserved, reported explicitly.

Reduced QAOA instance (3 EVs, 12 qubits = 6 decision + 6 slack, p=1, 512 shots, COBYLA, 34 evaluations,
circuit depth 17, 66 gates, ~1.5 s): the best feasible sample equals the brute-force optimum and CP-SAT
(5.785). Depth sweep p=1,2,3 (depth 17/31/45, gates 66/120/174) also reached the optimum.

## Performance comparison — honest reading

* CP-SAT is exact at these sizes and is never beaten. Greedy is already optimal on the uncongested default.
* **No quantum advantage is claimed or observed.** A control that draws the same number of *uniform random*
  bitstrings (~17 000) finds the same optimum on the 12-qubit instance. chance per sample of hitting the best plan is 1.9% for QAOA vs 1.5% for random. QAOA's final
  P(feasible) is ~10–16%, similar to random sampling's 12.5%, and deeper circuits did not raise it.
  The instance is simply small enough that sampling finds the optimum.
* The hybrid gives a valid schedule using QAOA for every batch decision. On the default scenario it is
  slightly worse than classical (batches are myopic); on the congested one it beats greedy and is close to CP-SAT.
* Value of the project: a correct QUBO/QAOA pipeline, validated end to end, ready for hardware-scale studies.

## Notes on scope

QAOA needs a cost Hamiltonian; the QUBO is the standard way to write one, so it cannot be dropped (the dashboard hides it and explains results in plain English). Equal values on the reduced instance are correct: all methods reach the true optimum. The simulator is noiseless, so no error correction is used.

## Why simulation, not real hardware

Hackday scope, reproducibility (seeded simulator, no queues, no credentials), and the instances are ≤18
qubits where exact simulation is cheap. Real devices would add noise without helping at this size. The
code uses only generic Qiskit circuits, so a backend swap would be the only change for later work.

## Known limitations

* Constant charging power (no taper), contiguous sessions, straight-line travel.
* Station capacity = ports only (no grid power cap).
* Slack-bit encoding inflates qubit counts; r=1 conflicts use slack-free pair penalties.
* QAOA parameters are heuristic (linear-ramp init, ≤60 iterations); the sampled-energy objective is noisy.
* Rolling-horizon batches are greedy in time and can miss global optima; refinement is single-EV local search.
* Penalty weights are fixed multiples of the max cost, not tuned.
* Quantum behaviour is simulated; runtime numbers are simulator time, not hardware time.
