"""
Parameter-recovery check for estimate_from_tedsd.py.

Generates a fake TEDS-D file from a chain with known parameters (same coding,
same LOS bands, a known wait effect), runs the estimator on it, and prints the
true values next to the estimates and their bootstrap intervals. If the
estimator can't recover a chain it generated itself, it can't be trusted on
real data.

Usage: python check_estimator.py [--n 30000] [--boot 100]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from estimate_from_tedsd import chain_summary

TRUE = {
    "Hispanic": np.array([[0.010, 0.060, 0.040, 0.20],
                          [0.040, 0.030, 0.035, 0.10],
                          [0.080, 0.020, 0.035, 0.00]]),
    "Non-Hispanic": np.array([[0.012, 0.050, 0.045, 0.20],
                              [0.050, 0.025, 0.040, 0.10],
                              [0.095, 0.015, 0.035, 0.00]]),
}
TRUE_LOG_OR = 0.08  # per week waited, applied to the first-stage dropout probability


def los_code(day):
    if day <= 30:
        return day
    for code, hi in zip(range(31, 37), [45, 60, 90, 120, 180, 365]):
        if day <= hi:
            return code
    return 37


def simulate_episodes(h, n, eth_code, rng):
    rows = []
    waits = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.5, 0.3, 0.1, 0.06, 0.04])
    wait_wk = np.array([0, 4 / 7, 11 / 7, 22.5 / 7, 6.0])[waits]
    for i in range(n):
        hh = h.copy()
        odds = hh[0, 1] / (1 - hh[0, 1]) * np.exp(TRUE_LOG_OR * wait_wk[i])
        hh[0, 1] = odds / (1 + odds)
        s, w = 0, 0
        while True:
            w += 1
            p = np.append(hh[s], max(0.0, 1 - hh[s].sum()))
            k = rng.choice(5, p=p / p.sum())
            if k < 3:
                break
            if k == 3:
                s += 1
        day = 7 * (w - 1) + rng.integers(1, 8)
        rows.append((6, 7, eth_code, [1, 2, 4][k], los_code(day), 1, waits[i]))
    return rows


def true_summary(h):
    """Chain summary averaged over the wait distribution, since waits change stage-1 dropout."""
    B, t = 0, 0
    for p_w, wk in zip([0.5, 0.3, 0.1, 0.06, 0.04], [0, 4 / 7, 11 / 7, 22.5 / 7, 6.0]):
        hh = h.copy()
        odds = hh[0, 1] / (1 - hh[0, 1]) * np.exp(TRUE_LOG_OR * wk)
        hh[0, 1] = odds / (1 + odds)
        s = chain_summary(hh)
        B, t = B + p_w * s["B"], t + p_w * s["mean_weeks"]
    return {"B": B, "mean_weeks": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="episodes per group")
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = simulate_episodes(TRUE["Hispanic"], args.n, 2, rng) + simulate_episodes(TRUE["Non-Hispanic"], args.n, 4, rng)
    tmp = tempfile.mkdtemp()
    csv = os.path.join(tmp, "fake_tedsd.csv")
    pd.DataFrame(rows, columns=["STFIPS", "SERVICES_D", "ETHNIC", "REASON", "LOS", "PSYPROB", "DAYWAIT"]).to_csv(csv, index=False)
    out = os.path.join(tmp, "fit.json")
    subprocess.run([sys.executable, "estimate_from_tedsd.py", csv, "--boot", str(args.boot), "--out", out], check=True)
    fit = json.load(open(out))
    print("\nRecovery (true vs estimate [95% CI]):")
    for g in TRUE:
        t = true_summary(TRUE[g])
        f = fit["fits"][g]
        for k, name in enumerate(["Completed", "Dropped out", "Other exit"]):
            lo, hi = f["absorption_ci"][0][k], f["absorption_ci"][1][k]
            ok = "ok" if lo <= t["B"][k] <= hi else "MISS"
            print(f"  {g:13s} {name:12s} true {t['B'][k]:.3f}  est {f['absorption'][k]:.3f} [{lo:.3f}, {hi:.3f}] {ok}")
        lo, hi = f["mean_weeks_ci"]
        ok = "ok" if lo <= t["mean_weeks"] <= hi else "MISS"
        print(f"  {g:13s} mean weeks   true {t['mean_weeks']:.2f}  est {f['mean_weeks']:.2f} [{lo:.2f}, {hi:.2f}] {ok}")
    we = fit["wait_effect"]
    print(f"  wait log-OR per week: true {TRUE_LOG_OR} (on stage-1 dropout), "
          f"est {we['log_or_per_week']:.3f} ± {1.96 * we['se']:.3f} (on any dropout; expected to be smaller)")


if __name__ == "__main__":
    main()
