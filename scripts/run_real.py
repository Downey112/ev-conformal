"""Real-data experiment on ChargePlace Scotland.

    python scripts/run_real.py --name paper    --seeds 3                      # paper's test period
    python scripts/run_real.py --name extended --seeds 3 --end 2025-10-26     # 14 more months
    python scripts/run_real.py --name tariff   --seeds 3 --test-start 2023-03-13 --end 2023-08-06
"""
import argparse
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evconf.data import build_panel  # noqa: E402
from evconf.realdata import Protocol, run_seed  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def _one(p, seed, name):
    t0 = time.time()
    res = run_seed(build_panel(), p, seed, log=lambda s: print(s, flush=True))
    np.savez_compressed(os.path.join(OUT, f"real_{name}_seed{seed}.npz"), **res)
    print(f"seed {seed} finished in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="paper")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--test-start", default=Protocol.test_start)
    ap.add_argument("--end", default=Protocol.end_all)
    args = ap.parse_args()
    build_panel()
    p = Protocol(test_start=args.test_start, end_all=args.end)
    Parallel(n_jobs=min(args.jobs, args.seeds))(delayed(_one)(p, s, args.name) for s in range(1, args.seeds + 1))


if __name__ == "__main__":
    main()
