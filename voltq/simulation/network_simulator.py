"""Deterministic event simulator: initial network -> events -> recovery strategies -> comparison."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..classical import cpsat_schedule, greedy_schedule
from ..data import generate_scenario
from ..hybrid import HybridConfig, hybrid_schedule
from ..models import ScenarioConfig
from ..preprocess import build_options
from ..qaoa import QAOAConfig
from ..recovery.commitments import RecoveryWeights, assign_commitments
from ..recovery.detection import AffectedReport, identify_affected
from ..recovery.events import Event, apply_events
from ..recovery.recovery_metrics import compute_metrics, metrics_frame
from ..recovery.recovery_optimizer import STRATEGIES, RecoveryResult, recover
from ..recovery.state import NetworkState


def build_initial_state(cfg: ScenarioConfig, solver: str = "CP-SAT", seed: int | None = None,
                        weights: RecoveryWeights | None = None,
                        qcfg: QAOAConfig | None = None) -> NetworkState:
    """Generate the network and its initial optimised schedule (Greedy, CP-SAT or hybrid QAOA)."""
    scn = generate_scenario(cfg)
    pre = build_options(scn)
    if solver.startswith("Greedy"):
        sched = greedy_schedule(scn, pre)
    elif solver.startswith("Hybrid"):
        sched = hybrid_schedule(scn, pre, qcfg or QAOAConfig(maxiter=40, shots=256), HybridConfig())[1]
    else:
        sched = cpsat_schedule(scn, pre)
    commitments = assign_commitments(scn, pre, sched.assignments, cfg.seed if seed is None else seed)
    return NetworkState(scn, pre.refs, dict(sched.assignments), commitments, 0,
                        weights=weights or RecoveryWeights())


@dataclass
class DisruptionRun:
    before: NetworkState
    after: NetworkState
    events: list[Event]
    report: AffectedReport
    notes: list[str]
    results: dict[str, RecoveryResult] = field(default_factory=dict)
    metrics: dict[str, dict] = field(default_factory=dict)


class NetworkSimulator:
    """Holds the current state; `trigger` applies events, `recover` runs strategies."""

    def __init__(self, state: NetworkState):
        self.state = state

    def trigger(self, events: list[Event]) -> DisruptionRun:
        eff = apply_events(self.state, events)
        report = identify_affected(eff.state, eff.directly_affected)
        return DisruptionRun(self.state, eff.state, events, report, eff.notes)

    def recover(self, run: DisruptionRun, strategies: tuple[str, ...] = STRATEGIES,
                qcfg: QAOAConfig | None = None, hcfg: HybridConfig | None = None,
                time_limit: float = 20.0) -> DisruptionRun:
        """Every strategy gets the identical post-event state, report, weights and constraints."""
        for name in strategies:
            res = recover(run.after, run.report.direct, name, qcfg=qcfg, hcfg=hcfg,
                          time_limit=time_limit, report=run.report)
            run.results[name] = res
            run.metrics[name] = compute_metrics(run.before, run.after, res, run.report)
        return run

    def commit(self, run: DisruptionRun, strategy: str) -> NetworkState:
        """Adopt a strategy's schedule as the new network state (for chained disruptions)."""
        res = run.results[strategy]
        new = run.after.copy()
        for i, o in res.assignments.items():
            old = run.before.assignments.get(i)
            if old is not None and (old.station_id, old.start) != (o.station_id, o.start):
                new.hist(i)["reschedules"] += 1
        new.assignments = dict(res.assignments)
        for e in new.scn.evs:
            if e.ev_id not in res.assignments:
                new.hist(e.ev_id)["delays"] += 1
        self.state = new
        return new

    @staticmethod
    def table(run: DisruptionRun):
        return metrics_frame(list(run.metrics.values()))
