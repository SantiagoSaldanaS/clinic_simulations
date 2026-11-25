"""
Fit the treatment Markov chain to TEDS-D by maximum likelihood.

Data: SAMHSA Treatment Episode Data Set, Discharges (TEDS-D) public use file, CSV:
  https://www.samhsa.gov/data/data-we-collect/teds-treatment-episode-data-set/datafiles
Each row is one treatment episode. We use its length of stay (LOS) and reason for
discharge (REASON), which together say when and how the episode left the chain.

Codes (TEDS-D codebook):
  STFIPS      6 = California
  SERVICES_D  6 = intensive outpatient, 7 = non-intensive outpatient
  ETHNIC      1, 2, 3, 5 = Hispanic or Latino; 4 = not Hispanic; -9 = missing
  REASON      1 completed, 2 dropped out, 3 terminated by facility, 4 transferred,
              5 incarcerated, 6 death, 7 other
  LOS         1-30 exact days; 31 = 31-45, 32 = 46-60, 33 = 61-90, 34 = 91-120,
              35 = 121-180, 36 = 181-365, 37 = more than 365
  PSYPROB     1 = co-occurring mental and substance use disorder
  DAYWAIT     0 = 0 days, 1 = 1-7, 2 = 8-14, 3 = 15-30, 4 = 31+

Method
  The chain has K transient stages in a row. Each week a patient in stage s
  completes, drops out, exits for another reason, advances to s+1, or stays.
  For any parameter value the chain gives the probability of leaving by each
  exit type in each week: f_k(w) = alpha Q^(w-1) R[:, k]. LOS is reported in days
  and in bands, so the likelihood of a row is the chain's probability of that
  exit type within that LOS interval (days assumed uniform within a week).
  We maximize the multinomial log-likelihood of the LOS-interval x exit-type
  table, pick K by BIC, get 95% intervals by resampling the table (bootstrap),
  and test whether Hispanic and non-Hispanic episodes share one chain with a
  likelihood-ratio test. The wait effect comes from a logistic regression of
  dropout on weeks waited (DAYWAIT), adjusted for ethnicity and co-occurring
  mental illness.

TEDS-D has ethnicity but no language variable, so the model's Spanish-preferring
patients use the Hispanic chain. Say so in the report.

Usage
  python estimate_from_tedsd.py tedsd_puf_2024.csv [--state 6] [--co-occurring]
                                 [--boot 200] [--out params_tedsd.json]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

LOS_INTERVALS = {**{d: (d, d) for d in range(1, 31)},
                 31: (31, 45), 32: (46, 60), 33: (61, 90), 34: (91, 120),
                 35: (121, 180), 36: (181, 365), 37: (366, None)}
LOS_CODES = list(LOS_INTERVALS)
MAX_WEEK = 260  # five years; the chain's leftover mass past this is negligible
DAYWAIT_WEEKS = {0: 0.0, 1: 4 / 7, 2: 11 / 7, 3: 22.5 / 7, 4: 6.0}  # band midpoints; 31+ set to 6 weeks
EXIT_NAMES = ["Completed", "Dropped out", "Other exit"]


def exit_type(reason):
    return np.select([reason == 1, reason == 2], [0, 1], default=2)


def interval_week_weights():
    """W[i, w]: share of week w+1's seven days that fall in LOS interval i."""
    W = np.zeros((len(LOS_CODES), MAX_WEEK))
    for i, code in enumerate(LOS_CODES):
        lo, hi = LOS_INTERVALS[code]
        hi = MAX_WEEK * 7 if hi is None else hi
        for day in range(lo, hi + 1):
            W[i, (day - 1) // 7] += 1 / 7
    return W


W_INTERVAL = interval_week_weights()


def unpack(theta, K):
    """Map unconstrained theta to stage rows [complete, drop, other, advance] via softmax."""
    rows, j = [], 0
    for s in range(K):
        n = 4 if s < K - 1 else 3
        z = np.append(theta[j:j + n], 0.0)  # last logit (stay) fixed at 0
        p = np.exp(z - z.max())
        p /= p.sum()
        rows.append(list(p[:3]) + [p[3] if s < K - 1 else 0.0])
        j += n
    return np.array(rows)


def n_params(K):
    return 4 * (K - 1) + 3


def exit_pmf(h):
    """f[w, k] = P(leave in week w+1 by exit k), starting in stage 1."""
    K = len(h)
    Q = np.zeros((K, K))
    for s in range(K):
        if s < K - 1:
            Q[s, s + 1] = h[s, 3]
        Q[s, s] = 1 - h[s].sum()
    R = h[:, :3]
    f = np.empty((MAX_WEEK, 3))
    v = np.zeros(K)
    v[0] = 1.0
    for w in range(MAX_WEEK):
        f[w] = v @ R
        v = v @ Q
    return f


def neg_loglik(theta, K, counts):
    f = exit_pmf(unpack(theta, K))
    p = np.clip(W_INTERVAL @ f, 1e-300, None)
    return -np.sum(counts * np.log(p))


def fit_chain(counts, K, start=None, restarts=4, seed=0):
    rng = np.random.default_rng(seed)
    starts = [start] if start is not None else []
    base = np.tile([-4.0, -3.0, -3.0, -1.5], K)[:n_params(K)]
    starts += [base + rng.normal(0, 0.5, n_params(K)) for _ in range(restarts)]
    best = None
    for s0 in starts:
        r = minimize(neg_loglik, s0, args=(K, counts), method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return best


def chain_summary(h):
    K = len(h)
    Q = np.zeros((K, K))
    for s in range(K):
        if s < K - 1:
            Q[s, s + 1] = h[s, 3]
        Q[s, s] = 1 - h[s].sum()
    N = np.linalg.inv(np.eye(K) - Q)
    return {"B": (N @ h[:, :3])[0], "mean_weeks": float((N @ np.ones(K))[0])}


def count_table(df):
    et = exit_type(df["REASON"].to_numpy())
    idx = pd.Index(LOS_CODES).get_indexer(df["LOS"].to_numpy())
    counts = np.zeros((len(LOS_CODES), 3))
    np.add.at(counts, (idx, et), 1)
    return counts


def fit_group(name, counts, Ks, boot, seed):
    N = counts.sum()
    print(f"\n{name}: {int(N)} episodes; observed shares C/D/O = "
          f"{np.round(counts.sum(0) / N * 100, 1)}")
    fits = {}
    for K in Ks:
        r = fit_chain(counts, K, seed=seed)
        bic = 2 * r.fun + n_params(K) * np.log(N)
        fits[K] = (r, bic)
        print(f"  K={K}: logLik {-r.fun:.1f}, BIC {bic:.1f}")
    K = min(fits, key=lambda k: fits[k][1])
    r = fits[K][0]
    h = unpack(r.x, K)
    summ = chain_summary(h)
    expected = (W_INTERVAL @ exit_pmf(h)) * N
    # Pearson check on the coarse table (LOS bands collapsed to 8 cells x 3 exits)
    bands = [range(0, 7), range(7, 14), range(14, 30), [30], [31], [32], [33], [34, 35, 36]]
    obs_c = np.array([counts[list(b)].sum(0) for b in bands])
    exp_c = np.array([expected[list(b)].sum(0) for b in bands])
    pearson = float(np.sum((obs_c - exp_c) ** 2 / np.maximum(exp_c, 1e-9)))

    rng = np.random.default_rng(seed + 1)
    p_hat = counts.ravel() / N
    boot_B, boot_t, boot_h = [], [], []
    t0 = time.time()
    for _ in range(boot):
        cb = rng.multinomial(int(N), p_hat).reshape(counts.shape).astype(float)
        rb = fit_chain(cb, K, start=r.x, restarts=0)
        hb = unpack(rb.x, K)
        sb = chain_summary(hb)
        boot_B.append(sb["B"]); boot_t.append(sb["mean_weeks"]); boot_h.append(hb)
    ci = lambda a: np.percentile(np.asarray(a), [2.5, 97.5], axis=0).tolist() if boot else None
    print(f"  chosen K={K}; absorption C/D/O = {np.round(summ['B'] * 100, 1)}; "
          f"mean weeks {summ['mean_weeks']:.1f}; Pearson X2 on 24 cells = {pearson:.1f}"
          f"  ({boot} bootstrap fits in {time.time() - t0:.0f}s)")
    return {
        "K": K, "stage_probs": np.round(h, 6).tolist(), "loglik": -r.fun, "n": int(N),
        "bic": {str(k): v[1] for k, v in fits.items()},
        "absorption": summ["B"].tolist(), "absorption_ci": ci(boot_B),
        "mean_weeks": summ["mean_weeks"], "mean_weeks_ci": ci(boot_t),
        "stage_probs_ci": ci(boot_h),
        "observed_shares": (counts.sum(0) / N).tolist(),
        "fit_check": {"pearson_x2": pearson, "observed": obs_c.tolist(), "expected": exp_c.tolist()},
        "theta": r.x.tolist(),
    }


def logistic(X, y):
    """Logistic regression by Newton steps; returns coefficients and standard errors."""
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.abs(np.diag(np.linalg.pinv(H))))  # pinv: rare dummy levels can make H near-singular
    return b, se


def wait_effect(df):
    d = df[df["DAYWAIT"].isin(DAYWAIT_WEEKS) & df["HISP"].notna()].copy()
    if len(d) < 500:
        return None
    wk = d["DAYWAIT"].map(DAYWAIT_WEEKS).to_numpy()
    X = np.column_stack([np.ones(len(d)), wk, d["HISP"].to_numpy(float),
                         (d["PSYPROB"] == 1).to_numpy(float)])
    y = (d["REASON"] == 2).to_numpy(float)
    keep = [0] + [j for j in range(1, X.shape[1]) if X[:, j].std() > 0]  # drop constant covariates
    b, se = logistic(X[:, keep], y)
    b = dict(zip(keep, b)); se = dict(zip(keep, se))
    table = d.assign(wk=wk, drop=y).groupby("DAYWAIT")["drop"].agg(["mean", "size"]).reset_index()
    out = {"n": int(len(d)), "log_or_per_week": float(b[1]), "se": float(se[1]),
           "or_per_week": float(np.exp(b[1])),
           "or_ci": [float(np.exp(b[1] - 1.96 * se[1])), float(np.exp(b[1] + 1.96 * se[1]))],
           "hispanic_log_or": float(b.get(2, np.nan)), "by_daywait": table.to_dict(orient="records")}
    print(f"\nWait effect (n={out['n']}): OR per week waited = {out['or_per_week']:.3f} "
          f"(95% CI {out['or_ci'][0]:.3f} to {out['or_ci'][1]:.3f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--state", type=int, default=6, help="STFIPS code; 6 = California, 0 = all states")
    ap.add_argument("--co-occurring", action="store_true", help="keep only PSYPROB == 1")
    ap.add_argument("--stages", default="2,3,4", help="K values to compare by BIC")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default="params_tedsd.json")
    args = ap.parse_args()

    want = {"STFIPS", "SERVICES_D", "ETHNIC", "REASON", "LOS", "PSYPROB", "DAYWAIT"}
    df = pd.read_csv(args.csv, usecols=lambda c: c.upper() in want)
    df.columns = [c.upper() for c in df.columns]
    n_all = len(df)
    if args.state:
        df = df[df["STFIPS"] == args.state]
    df = df[df["SERVICES_D"].isin([6, 7]) & df["LOS"].isin(LOS_CODES) & (df["REASON"] > 0)]
    if args.co_occurring:
        df = df[df["PSYPROB"] == 1]
    df = df.assign(HISP=np.select([df["ETHNIC"].isin([1, 2, 3, 5]), df["ETHNIC"] == 4], [1.0, 0.0], np.nan))
    print(f"{n_all} rows read; {len(df)} outpatient episodes kept "
          f"(STFIPS={args.state or 'all'}, co_occurring_only={args.co_occurring})")
    print(f"Hispanic share among episodes with known ethnicity: {np.nanmean(df['HISP']):.3f}")

    Ks = [int(k) for k in args.stages.split(",")]
    groups = {"Hispanic": df[df["HISP"] == 1], "Non-Hispanic": df[df["HISP"] == 0]}
    tables = {g: count_table(d) for g, d in groups.items()}
    res = {g: fit_group(g, tables[g], Ks, args.boot, seed=i) for i, g in enumerate(tables)}

    # Same K for both groups so the LRT compares nested models
    K = max(res[g]["K"] for g in res)
    sep = sum(fit_chain(tables[g], K, seed=7).fun for g in tables)
    pooled = fit_chain(tables["Hispanic"] + tables["Non-Hispanic"], K, seed=7).fun
    lr = 2 * (pooled - sep)
    p_lr = float(chi2.sf(lr, n_params(K)))
    print(f"\nLR test, one chain for both groups vs separate (K={K}): "
          f"LR = {lr:.1f}, df = {n_params(K)}, p = {p_lr:.3g}")

    out = {
        "note": f"MLE fit to {os.path.basename(args.csv)}; STFIPS={args.state or 'all'}; outpatient (SERVICES_D 6-7); "
                f"co_occurring_only={args.co_occurring}. Spanish-preferring patients use the Hispanic chain.",
        "stage_probs": {"Hispanic": res["Hispanic"]["stage_probs"], "Non-Hispanic": res["Non-Hispanic"]["stage_probs"]},
        "hispanic_share": float(np.nanmean(df["HISP"])),
        "fits": res,
        "lr_test": {"K": K, "lr": lr, "df": n_params(K), "p": p_lr},
    }
    we = wait_effect(df)
    if we:
        out["wait_effect"] = we
        out["wait_log_or_per_week"] = max(0.0, we["log_or_per_week"])
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {args.out}. Next: python sim.py --params {args.out}")


if __name__ == "__main__":
    main()
