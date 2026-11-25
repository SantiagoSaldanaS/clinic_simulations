"""
Language access and outpatient treatment outcomes: a stochastic clinic model.

Three layers, one per course module:
  1. Referrals arrive as Poisson processes, one per patient type.
  2. They wait for a caseload slot. Spanish-preferring patients need a bilingual
     clinician; everyone else can take any open slot. People who wait too long
     give up (reneging). Clinicians quit at random and their posts stay empty
     until someone is hired, which takes longer for bilingual posts.
  3. Admitted patients follow a weekly absorbing Markov chain fitted to
     California outpatient episodes in TEDS-D 2024, ending in completion,
     dropout, or another exit. Waiting raises the odds of dropping out during
     the first four weeks in care.

Usage:
  python sim.py                      # settings.py, chain from outputs/params_tedsd_ca.json if present
  python sim.py --reps 60 --out outputs
"""
import argparse
import json
import os
from collections import deque

import numpy as np

from settings import EARLY_WEEKS, load_settings

TYPES = ["Latino, Spanish-preferring", "Latino, English-proficient", "Non-Latino"]
CHAIN_OF_TYPE = ["Hispanic", "Hispanic", "Non-Hispanic"]
NEEDS_BILINGUAL = [True, False, False]
COMPLETED, DROPPED, OTHER, RENEGED, UNRESOLVED = 0, 1, 2, 3, 4
OUTCOMES = ["Completed", "Dropped out", "Other exit", "Gave up waiting", "Unresolved"]


def chain_matrices(h):
    h = np.asarray(h, float)
    K = len(h)
    Q = np.zeros((K, K))
    for s in range(K):
        if s < K - 1:
            Q[s, s + 1] = h[s, 3]
        Q[s, s] = 1 - h[s].sum()
    return Q, h[:, :3]


def absorbing_analysis(h):
    """N = (I-Q)^-1, absorption probabilities B = NR, expected weeks t = N1."""
    Q, R = chain_matrices(h)
    N = np.linalg.inv(np.eye(len(Q)) - Q)
    return {"Q": Q, "R": R, "N": N, "B": N @ R, "t": N @ np.ones(len(Q))}


def arrival_rates(p):
    """Little's law: total rate = offered_load * slots / mean weeks in treatment, split by type shares."""
    shares = np.array([p["latino_share"] * p["spanish_pref_share"],
                       p["latino_share"] * (1 - p["spanish_pref_share"]),
                       1 - p["latino_share"]])
    t = np.array([absorbing_analysis(p["stage_probs"][c])["t"][0] for c in CHAIN_OF_TYPE])
    slots = (p["general_clinicians"] + p["bilingual_clinicians"]) * p["caseload_per_clinician"]
    total = p["offered_load"] * slots / float(shares @ t)
    return total * shares


def simulate(p, seed=0, warmup=104, cohort_weeks=104, followup=156, record=False):
    """
    Weekly-clock simulation. Returns one record per patient referred during the
    cohort window. Separate RNG streams for arrivals, reneging and treatment keep
    random numbers common across scenarios, which sharpens comparisons.
    """
    rng_arr, rng_ren, rng_tx, rng_staff = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    chains = [np.asarray(p["stage_probs"][c], float) for c in ("Hispanic", "Non-Hispanic")]
    Kmax = max(len(c) for c in chains)
    probs = np.zeros((2, Kmax, 5))  # chain, stage, [complete, drop, other, advance, stay]
    for g, h in enumerate(chains):
        probs[g, :len(h), :4] = h
        probs[g, :, 4] = 1 - probs[g, :, :4].sum(axis=1)
    chain_idx = [0, 0, 1]
    lam = arrival_rates(p)
    gamma, renege = p["wait_log_or_per_week"], p["renege_prob"]
    K = p["caseload_per_clinician"]
    staff = [p["general_clinicians"], p["bilingual_clinicians"]]  # posts per pool: 0 general, 1 bilingual
    vacant = [[0] * staff[0], [0] * staff[1]]  # weeks left until each post is filled again (0 = filled)
    quit_p = 1 - np.exp(-p.get("turnover_annual", 0.0) / 52)
    fill_weeks = [p.get("vacancy_weeks_general", 8.7),
                  p.get("vacancy_weeks_general", 8.7) * p.get("bilingual_vacancy_factor", 1.5)]
    cap = [staff[0] * K, staff[1] * K]
    spill = p.get("bilingual_spillover", True)
    horizon = warmup + cohort_weeks + followup

    ptype, parrive, padmit, pout = [], [], [], []
    queues = [deque(), deque()]  # 0: needs bilingual, 1: any clinician
    a_id = np.empty(0, int); a_c = np.empty(0, int); a_s = np.empty(0, int)
    a_odds = np.empty(0); a_pool = np.empty(0, int); a_wk = np.empty(0, int)
    used = [0, 0]
    log = []

    for week in range(horizon):
        for pool in (0, 1):
            for j in range(staff[pool]):
                if vacant[pool][j] > 0:
                    vacant[pool][j] -= 1
                elif rng_staff.random() < quit_p:
                    vacant[pool][j] = rng_staff.geometric(1 / fill_weeks[pool])
            # A departing clinician's patients stay in care; new admissions to that pool wait for the hire
            cap[pool] = K * sum(v == 0 for v in vacant[pool])

        for t in range(3):
            for _ in range(rng_arr.poisson(lam[t])):
                pid = len(ptype)
                ptype.append(t); parrive.append(week); padmit.append(-1); pout.append(UNRESOLVED)
                queues[0 if NEEDS_BILINGUAL[t] else 1].append(pid)

        admitted, pools = [], []
        while queues[0] and used[1] < cap[1]:
            admitted.append(queues[0].popleft()); pools.append(1); used[1] += 1
        while queues[1] and used[0] < cap[0]:
            admitted.append(queues[1].popleft()); pools.append(0); used[0] += 1
        # Spare bilingual slots go to the general queue only when no Spanish-preferring patient waits
        while spill and queues[1] and not queues[0] and used[1] < cap[1]:
            admitted.append(queues[1].popleft()); pools.append(1); used[1] += 1
        if admitted:
            ids = np.array(admitted)
            waits = np.minimum(np.array([week - parrive[i] for i in admitted], float), 12)
            for i in admitted:
                padmit[i] = week
            a_id = np.concatenate([a_id, ids])
            a_c = np.concatenate([a_c, [chain_idx[ptype[i]] for i in admitted]])
            a_s = np.concatenate([a_s, np.zeros(len(ids), int)])
            a_odds = np.concatenate([a_odds, np.exp(gamma * waits)])
            a_pool = np.concatenate([a_pool, pools])
            a_wk = np.concatenate([a_wk, np.zeros(len(ids), int)])

        for q in (0, 1):
            if queues[q]:
                stay = rng_ren.random(len(queues[q])) >= renege
                kept = deque()
                for pid, s in zip(queues[q], stay):
                    if s:
                        kept.append(pid)
                    else:
                        pout[pid] = RENEGED
                queues[q] = kept

        if len(a_id):
            P = probs[a_c, a_s].copy()
            early = a_wk < EARLY_WEEKS
            d = P[early, 1]
            d_new = d * a_odds[early] / (1 - d + d * a_odds[early])  # waiting multiplies this week's dropout odds
            P[early] *= ((1 - d_new) / (1 - d))[:, None]
            P[early, 1] = d_new
            cum = np.cumsum(P / P.sum(axis=1, keepdims=True), axis=1)
            choice = (rng_tx.random(len(a_id))[:, None] > cum).sum(axis=1)
            done = choice <= 2
            for pid, c, pool in zip(a_id[done], choice[done], a_pool[done]):
                pout[pid] = int(c)
                used[pool] -= 1
            a_s = np.where(choice == 3, a_s + 1, a_s)
            a_wk = a_wk + 1
            keep = ~done
            a_id, a_c, a_s, a_odds, a_pool, a_wk = a_id[keep], a_c[keep], a_s[keep], a_odds[keep], a_pool[keep], a_wk[keep]

        if record:
            log.append([week, len(queues[0]), len(queues[1]), used[0], used[1], cap[0] // K, cap[1] // K])

    rec = {k: np.array(v) for k, v in zip(("type", "arrive", "admit", "outcome"), (ptype, parrive, padmit, pout))}
    window = (rec["arrive"] >= warmup) & (rec["arrive"] < warmup + cohort_weeks)
    rec = {k: v[window] for k, v in rec.items()}
    if record:
        rec["log"] = np.array(log)
    return rec


def summarize(rec):
    out = {}
    for t, name in enumerate(TYPES):
        m = rec["type"] == t
        oc = rec["outcome"][m]
        adm = m & (rec["admit"] >= 0)
        waited = m & ((rec["admit"] < 0) | (rec["admit"] > rec["arrive"]))  # ever on the waitlist
        out[name] = {
            "referred": int(m.sum()),
            "pct_completed": 100 * np.mean(oc == COMPLETED) if m.any() else np.nan,
            "pct_reneged": 100 * np.mean(oc == RENEGED) if m.any() else np.nan,
            "pct_unresolved": 100 * np.mean(oc == UNRESOLVED) if m.any() else np.nan,
            "mean_wait_weeks": float(np.mean(rec["admit"][adm] - rec["arrive"][adm])) if adm.any() else np.nan,
            "pct_never_admitted_if_waitlisted": 100 * np.mean(rec["admit"][waited] < 0) if waited.any() else np.nan,
            "completed_per_year": float(np.sum(oc == COMPLETED)) / 2,  # cohort window is two years
            "shares_if_admitted": [100 * np.mean(rec["outcome"][adm] == k) for k in (COMPLETED, DROPPED, OTHER)]
            if adm.any() else [np.nan] * 3,
        }
    sp, nl = out[TYPES[0]]["pct_completed"], out[TYPES[2]]["pct_completed"]
    out["gap_pp"] = nl - sp
    out["total_completed_per_year"] = sum(out[n]["completed_per_year"] for n in TYPES)
    return out


def replicate(p, reps, seed0=1000):
    return [summarize(simulate(p, seed=seed0 + r)) for r in range(reps)]


def mean_ci(values):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan, np.nan)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))


def collect(res, name, key):
    return mean_ci([r[name][key] if name else r[key] for r in res])


COMMUNITIES = [
    # name, Latino share of referrals (TEDS-D 2024 CA outpatient), Spanish-preferring share of Latinos
    ("Sacramento metro", 0.267, 0.27),
    ("California", 0.476, 0.27),
    ("Los Angeles metro", 0.555, 0.27),
    ("Salinas metro", 0.648, 0.27),
    ("San Benito County profile", 0.64, 0.39),
    ("El Centro metro", 0.860, 0.39),
]


def staffing_by_community(p, reps, bilingual_options=(0, 1, 2)):
    """For a clinic of fixed size, how does the bilingual headcount change outcomes in each community?"""
    rows = []
    total = p["general_clinicians"] + p["bilingual_clinicians"]
    for name, ls, sp in COMMUNITIES:
        for b in bilingual_options:
            q = dict(p, latino_share=ls, spanish_pref_share=sp, bilingual_clinicians=b, general_clinicians=total - b)
            res = replicate(q, reps)
            row = {"community": name, "latino_share": ls, "spanish_pref_share": sp,
                   "spanish_pref_of_referrals": ls * sp, "bilingual": b,
                   "gap_pp": collect(res, None, "gap_pp"),
                   "total_completed": collect(res, None, "total_completed_per_year")}
            for n in TYPES:
                for k in ("pct_completed", "pct_reneged", "mean_wait_weeks"):
                    row[f"{n}|{k}"] = collect(res, n, k)
            rows.append(row)
            print(f"  {name:26s} {b} bilingual of {total}: Spanish-pref completed "
                  f"{row[TYPES[0] + '|pct_completed'][0]:5.1f}% vs non-Latino {row[TYPES[2] + '|pct_completed'][0]:5.1f}%, "
                  f"wait {row[TYPES[0] + '|mean_wait_weeks'][0]:.1f} wk, gave up {row[TYPES[0] + '|pct_reneged'][0]:.0f}%")
    return rows


def spillover_policy(p, reps):
    """Same staff, two rules for idle bilingual slots: lend them to anyone, or hold them for Spanish speakers."""
    out = {}
    for spill in (True, False):
        res = replicate(dict(p, bilingual_spillover=spill), reps)
        out["lend" if spill else "reserve"] = {
            "gap_pp": collect(res, None, "gap_pp"),
            "total_completed": collect(res, None, "total_completed_per_year"),
            **{f"{n}|{k}": collect(res, n, k) for n in TYPES for k in ("pct_completed", "mean_wait_weeks")},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", help="chain JSON from estimate_from_tedsd.py (default: outputs/params_tedsd_ca.json)")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    p, meta = load_settings(args.params)
    os.makedirs(args.out, exist_ok=True)
    np.set_printoptions(precision=4, suppress=True)
    print(f"Chain source: {meta['chain_source']}")

    analytic = {}
    for c in ("Hispanic", "Non-Hispanic"):
        a = absorbing_analysis(p["stage_probs"][c])
        analytic[c] = {"B": a["B"][0].tolist(), "mean_weeks": float(a["t"][0]), "N": a["N"].tolist()}
        print(f"[{c}] K={len(a['Q'])} stages; absorption from stage 1 (C, D, O) = {np.round(a['B'][0] * 100, 1)}; "
              f"expected weeks = {a['t'][0]:.1f}")
    lam = arrival_rates(p)
    print("Arrival rates per week:", dict(zip(TYPES, np.round(lam, 3))))

    # With no waiting and unlimited slots the simulation must reproduce B = NR
    val = dict(p, wait_log_or_per_week=0.0, general_clinicians=40, bilingual_clinicians=40, offered_load=0.3)
    vres = replicate(val, 40)
    validation = {}
    for t, c in [(0, "Hispanic"), (2, "Non-Hispanic")]:
        sim_b = np.nanmean([r[TYPES[t]]["shares_if_admitted"] for r in vres], axis=0)
        validation[c] = {"analytic": (100 * np.array(analytic[c]["B"])).tolist(), "simulated": sim_b.tolist()}
        print(f"Check [{c}]: analytic {np.round(100 * np.array(analytic[c]['B']), 1)} vs simulated {np.round(sim_b, 1)}")

    print("\nBaseline:")
    base = replicate(p, args.reps)
    baseline = {n: {k: collect(base, n, k) for k in ("pct_completed", "pct_reneged", "mean_wait_weeks",
                                                       "pct_never_admitted_if_waitlisted", "pct_unresolved")}
                for n in TYPES}
    baseline["gap_pp"] = collect(base, None, "gap_pp")
    for n in TYPES:
        b = baseline[n]
        print(f"  {n:28s} completed {b['pct_completed'][0]:5.1f}%  gave up {b['pct_reneged'][0]:5.1f}%  "
              f"wait {b['mean_wait_weeks'][0]:.2f} wk  unresolved {b['pct_unresolved'][0]:.1f}%")

    print("\nStaffing by community (3 clinicians):")
    staffing = staffing_by_community(p, args.reps)
    print("\nIdle bilingual slots: lend vs reserve (San Benito profile, 1 bilingual of 3):")
    policy = spillover_policy(dict(p, latino_share=0.64, spanish_pref_share=0.39), args.reps)
    for k, v in policy.items():
        print(f"  {k:8s} gap {v['gap_pp'][0]:.1f} pp, completions/yr {v['total_completed'][0]:.1f}")

    run = simulate(dict(p, latino_share=0.64, spanish_pref_share=0.39), seed=11, record=True)["log"]
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"settings": p, "meta": meta, "arrival_rates": dict(zip(TYPES, lam.tolist())),
                   "analytic": analytic, "validation": validation, "baseline": baseline,
                   "staffing": staffing, "policy": policy, "trace": run.tolist()}, f, indent=1)
    print(f"\nSaved {args.out}/results.json")


if __name__ == "__main__":
    main()
