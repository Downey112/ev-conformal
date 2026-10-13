"""Simulation study: coverage of CP / CP-MNR / CP-MNR-Nested* as a function of
(a) how strongly activity depends on demand and (b) calibration size.

    python scripts/run_sim.py --reps 300
"""
import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evconf.simulate import Config, Hierarchy, one_replication, summarise  # noqa: E402


def run_config(cfg, reps, seed, n_jobs):
    h = Hierarchy(cfg)
    seeds = np.random.SeedSequence(seed).spawn(reps)
    out = Parallel(n_jobs=n_jobs)(delayed(one_replication)(h, s) for s in seeds)
    return summarise(h, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "sim.csv"))
    args = ap.parse_args()

    base = Config()
    grid = [("rho", replace(base, rho=r)) for r in (0.0, 0.3, 0.6, 0.9)]
    grid += [("n_cal", replace(base, rho=0.6, n_cal=n)) for n in (25, 50, 100, 400)]

    frames = []
    for i, (sweep, cfg) in enumerate(grid):
        t0 = time.time()
        df = pd.DataFrame(run_config(cfg, args.reps, seed=1000 + i, n_jobs=args.jobs))
        df.insert(0, "n_cal", cfg.n_cal)
        df.insert(0, "rho", cfg.rho)
        df.insert(0, "sweep", sweep)
        frames.append(df)
        print(f"{sweep}: rho={cfg.rho} n_cal={cfg.n_cal} done in {time.time() - t0:.0f}s", flush=True)
        pd.concat(frames).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
