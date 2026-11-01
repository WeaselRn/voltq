"""Multi-seed recovery benchmark -> outputs/recovery_benchmark.csv + printed summary.

    python run_recovery_benchmark.py [n_seeds] [n_evs]
"""
import sys
from pathlib import Path

from voltq.qaoa import QAOAConfig
from voltq.simulation.benchmark import run_benchmark, summarize

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n_evs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    df = run_benchmark(seeds=tuple(range(1, n_seeds + 1)), n_evs=n_evs,
                       qcfg=QAOAConfig(maxiter=30, shots=256))
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "recovery_benchmark.csv", index=False)
    print(f"{len(df)} rows, {df['valid'].mean() * 100:.0f}% valid schedules (independent validator)")
    print(summarize(df).round(2).to_string())
    print("\nmean recovery objective by scenario (lower = better)")
    print(df.pivot_table(index="scenario", columns="strategy", values="recovery_objective").round(1).to_string())
