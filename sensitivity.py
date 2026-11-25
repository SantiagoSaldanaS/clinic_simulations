"""
Sensitivity of the main results to every setting that isn't estimated from TEDS-D.

One-way: move each setting to its low and high value (settings.RANGES) with the
rest at base. Global: Latin hypercube over all ranges at once, then partial rank
correlation (PRCC) between each setting and the outcome.

Reference case: San Benito County profile (64% Latino, 39% of them Spanish-
preferring), three clinicians, one bilingual. Outcomes: completion gap between
non-Latino and Spanish-preferring referrals, and the share of Spanish-preferring
referrals who give up before a first appointment.

Usage: python sensitivity.py [--reps 100] [--lhs 300] [--lhs-reps 20]
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import qmc, rankdata

from settings import RANGES_KEYS, load_settings
from sim import TYPES, mean_ci, replicate

INTEGER = {"caseload_per_clinician"}


def outcome(res):
    gap = mean_ci([r["gap_pp"] for r in res])
    gave_up = mean_ci([r[TYPES[0]]["pct_reneged"] for r in res])
    return gap, gave_up


def prcc(X, y):
    R = np.column_stack([rankdata(c) for c in X.T])
    ry = rankdata(y)
    out = []
    for j in range(R.shape[1]):
        others = np.column_stack([np.ones(len(R)), np.delete(R, j, axis=1)])
        rx = R[:, j] - others @ np.linalg.lstsq(others, R[:, j], rcond=None)[0]
        rr = ry - others @ np.linalg.lstsq(others, ry, rcond=None)[0]
        out.append(float(np.corrcoef(rx, rr)[0, 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--lhs", type=int, default=300)
    ap.add_argument("--lhs-reps", type=int, default=20)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    p, meta = load_settings()
    ref = dict(p, latino_share=0.64, spanish_pref_share=0.39)
    ranges = meta["ranges"]

    base_gap, base_gu = outcome(replicate(ref, args.reps))
    print(f"Reference: gap {base_gap[0]:.1f} ± {base_gap[1]:.1f} pp; Spanish-pref gave up {base_gu[0]:.1f} ± {base_gu[1]:.1f}%")
    oneway = {}
    for k in RANGES_KEYS:
        lo, hi, _ = ranges[k]
        vals = {}
        for label, v in (("low", lo), ("high", hi)):
            v = int(round(v)) if k in INTEGER else v
            g, gu = outcome(replicate(dict(ref, **{k: v}), args.reps))
            vals[label] = {"value": v, "gap": g, "gave_up": gu}
        oneway[k] = vals
        print(f"  {k:26s} low {vals['low']['value']:>7}: gap {vals['low']['gap'][0]:5.1f}, gave up {vals['low']['gave_up'][0]:5.1f}%"
              f" | high {vals['high']['value']:>7}: gap {vals['high']['gap'][0]:5.1f}, gave up {vals['high']['gave_up'][0]:5.1f}%")

    sampler = qmc.LatinHypercube(d=len(RANGES_KEYS), seed=3)
    U = sampler.random(args.lhs)
    X = np.empty_like(U)
    gaps, gus = [], []
    for i, u in enumerate(U):
        q = dict(ref)
        for j, k in enumerate(RANGES_KEYS):
            lo, hi, _ = ranges[k]
            v = lo + u[j] * (hi - lo)
            q[k] = int(round(v)) if k in INTEGER else v
            X[i, j] = q[k]
        g, gu = outcome(replicate(q, args.lhs_reps, seed0=5000 + 100 * i))
        gaps.append(g[0]); gus.append(gu[0])
    gaps, gus = np.array(gaps), np.array(gus)
    global_res = {
        "n_samples": args.lhs, "reps_each": args.lhs_reps,
        "gap_quantiles": np.percentile(gaps, [5, 25, 50, 75, 95]).tolist(),
        "share_gap_over_5pp": float(np.mean(gaps > 5)),
        "gave_up_quantiles": np.percentile(gus, [5, 25, 50, 75, 95]).tolist(),
        "prcc_gap": dict(zip(RANGES_KEYS, prcc(X, gaps))),
        "prcc_gave_up": dict(zip(RANGES_KEYS, prcc(X, gus))),
    }
    print(f"\nGlobal ({args.lhs} samples): gap 5-95% = {global_res['gap_quantiles'][0]:.1f} to "
          f"{global_res['gap_quantiles'][4]:.1f} pp; gap > 5 pp in {global_res['share_gap_over_5pp']:.0%} of samples")
    for k in RANGES_KEYS:
        print(f"  PRCC {k:26s} gap {global_res['prcc_gap'][k]:+.2f}   gave up {global_res['prcc_gave_up'][k]:+.2f}")
    with open(os.path.join(args.out, "sensitivity.json"), "w") as f:
        json.dump({"reference": {"gap": base_gap, "gave_up": base_gu}, "oneway": oneway, "global": global_res,
                   "lhs_samples": {"X": X.tolist(), "keys": RANGES_KEYS, "gap": gaps.tolist(), "gave_up": gus.tolist()}},
                  f, indent=1)
    print(f"Saved {args.out}/sensitivity.json")


if __name__ == "__main__":
    main()
