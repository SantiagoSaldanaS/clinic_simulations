"""
Location-specific settings for California metros, from TEDS-D 2024.

For each metro (CBSA) with enough outpatient episodes:
  - Latino share of episodes
  - its own treatment chain per group, fitted by maximum likelihood with the
    statewide number of phases (falls back to the statewide chain when a group
    has fewer than MIN_GROUP episodes)
  - the share of admitted episodes that waited 8+ days (DAYWAIT >= 2)
  - a referral volume calibrated so the simulated clinic reproduces that wait share

Spanish-preferring share stays at the statewide 27%: no source we found splits
English proficiency among Latinos by county.

Usage: python communities.py tedsd_puf_2024.csv [--reps 40]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from estimate_from_tedsd import LOS_CODES, chain_summary, count_table, fit_chain, unpack
from settings import load_settings
from sim import simulate

METROS = {  # CBSA 2020 code: name
    31080: "Los Angeles", 40140: "Riverside-San Bernardino", 41740: "San Diego", 40900: "Sacramento",
    41860: "San Francisco-Oakland", 41940: "San Jose", 23420: "Fresno", 12540: "Bakersfield",
    41500: "Salinas", 47300: "Visalia", 37100: "Oxnard", 33700: "Modesto", 32900: "Merced",
    20940: "El Centro",
}
MIN_GROUP = 400


def wait_share_sim(p, reps, seed0=7000):
    """Share of admitted referrals who waited one week or more in the simulated clinic."""
    num = den = 0
    for r in range(reps):
        rec = simulate(p, seed=seed0 + r)
        adm = rec["admit"] >= 0
        num += np.sum(rec["admit"][adm] - rec["arrive"][adm] >= 1)
        den += adm.sum()
    return num / max(den, 1)


def calibrate_load(p, target, reps):
    f = lambda load: wait_share_sim(dict(p, offered_load=load), reps) - target
    lo, hi = 0.4, 1.6
    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return lo, "target below the model's floor; set to the lowest load tested"
    if fhi < 0:
        return hi, "target above what the model reaches; set to the highest load tested"
    return brentq(f, lo, hi, xtol=0.01), "calibrated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default="outputs/communities.json")
    args = ap.parse_args()
    base, meta = load_settings()
    state = json.load(open(os.path.join("outputs", "params_tedsd_ca.json")))
    df = pd.read_csv(args.csv, usecols=["STFIPS", "SERVICES_D", "ETHNIC", "REASON", "LOS", "DAYWAIT", "CBSA2020"])
    ca = df[(df["STFIPS"] == 6) & df["SERVICES_D"].isin([6, 7]) & df["LOS"].isin(LOS_CODES) & (df["REASON"] > 0)].copy()
    ca["HISP"] = np.select([ca["ETHNIC"].isin([1, 2, 3, 5]), ca["ETHNIC"] == 4], [1.0, 0.0], np.nan)

    def profile(name, d, code=None):
        out = {"name": name, "cbsa": code, "n": int(len(d)), "latino_share": round(float(np.nanmean(d["HISP"])), 3),
               "spanish_pref_share": 0.27, "stage_probs": {}, "chain_source": {}, "observed": {}}
        for g, flag in (("Hispanic", 1.0), ("Non-Hispanic", 0.0)):
            sub = d[d["HISP"] == flag]
            f = state["fits"][g]
            if code is None or len(sub) < MIN_GROUP:
                h = np.array(state["stage_probs"][g])
                out["chain_source"][g] = "statewide" if code is None else f"statewide (only {len(sub)} episodes here)"
            else:
                r = fit_chain(count_table(sub), f["K"], start=np.array(f["theta"]), restarts=0)
                h = unpack(r.x, f["K"])
                out["chain_source"][g] = f"fitted to {len(sub)} episodes"
            out["stage_probs"][g] = np.round(h, 6).tolist()
            s = chain_summary(h)
            out["observed"][g] = {"n": int(len(sub)), "completed": round(float((sub["REASON"] == 1).mean()), 3),
                                  "dropped_out": round(float((sub["REASON"] == 2).mean()), 3),
                                  "chain_completed": round(float(s["B"][0]), 3), "mean_weeks": round(s["mean_weeks"], 1)}
        known = d[d["DAYWAIT"] >= 0]
        out["wait_8plus_days"] = round(float((known["DAYWAIT"] >= 2).mean()), 4)
        p = dict(base, latino_share=out["latino_share"], spanish_pref_share=0.27, stage_probs=out["stage_probs"])
        out["offered_load"], out["load_note"] = calibrate_load(p, out["wait_8plus_days"], args.reps)
        out["offered_load"] = round(out["offered_load"], 3)
        o = out["observed"]
        print(f"{name:24s} n={out['n']:6d} Latino {out['latino_share']:.0%}  completed H {o['Hispanic']['completed']:.0%} "
              f"/ NH {o['Non-Hispanic']['completed']:.0%}  waited 8+ days {out['wait_8plus_days']:.1%}  "
              f"-> load {out['offered_load']:.2f} ({out['load_note']})", flush=True)
        return out

    rows = [profile("California", ca)]
    for code, name in METROS.items():
        d = ca[ca["CBSA2020"] == code]
        if len(d) >= 500:
            rows.append(profile(name, d, code))
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
