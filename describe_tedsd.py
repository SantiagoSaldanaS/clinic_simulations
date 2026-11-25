"""
Descriptive and adjusted analysis of California outpatient episodes in TEDS-D,
used to set and check the simulation's inputs.

Usage: python describe_tedsd.py tedsd_puf_2024.csv [--out outputs/tedsd_ca_summary.json]
"""
import argparse
import json

import numpy as np
import pandas as pd

from estimate_from_tedsd import logistic

COLS = ["STFIPS", "SERVICES_D", "ETHNIC", "REASON", "LOS", "PSYPROB", "DAYWAIT", "AGE", "SEX",
        "SUB1", "PSOURCE", "NOPRIOR", "CBSA2020", "PRIMPAY"]
ORIGIN = {1: "Puerto Rican", 2: "Mexican", 3: "Cuban or other specific", 5: "Not specified"}


def pct(x):
    return round(100 * float(x), 1)


def dummies(df, col, ref):
    vals = sorted(v for v in df[col].unique() if v != ref)
    return np.column_stack([(df[col] == v).to_numpy(float) for v in vals]), [f"{col}={v}" for v in vals]


def adjusted_or(df, outcome):
    """Odds ratio for Hispanic vs non-Hispanic, adjusted for case mix."""
    d = df[df["HISP"].notna()].copy()
    for c in ["AGE", "SEX", "SUB1", "PSOURCE", "NOPRIOR", "PSYPROB", "SERVICES_D"]:
        d[c] = d[c].where(d[c] > 0, -9)  # keep missing as its own level
    blocks, names = [np.ones((len(d), 1)), d[["HISP"]].to_numpy(float)], ["const", "HISP"]
    for c, ref in [("AGE", 5), ("SEX", 1), ("SUB1", 2), ("PSOURCE", 1), ("NOPRIOR", 0),
                   ("PSYPROB", 2), ("SERVICES_D", 7)]:
        X, n = dummies(d, c, ref)
        keep = X.sum(0) >= 50  # drop rare levels so the fit stays stable
        blocks.append(X[:, keep]); names += [nm for nm, k in zip(n, keep) if k]
    X = np.column_stack(blocks)
    y = outcome(d).to_numpy(float)
    b, se = logistic(X, y)
    return {"or": float(np.exp(b[1])), "ci": [float(np.exp(b[1] - 1.96 * se[1])), float(np.exp(b[1] + 1.96 * se[1]))],
            "n": int(len(d)), "covariates": names[2:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="outputs/tedsd_ca_summary.json")
    args = ap.parse_args()
    df = pd.read_csv(args.csv, usecols=COLS)
    ca = df[(df["STFIPS"] == 6) & df["SERVICES_D"].isin([6, 7]) & (df["REASON"] > 0)].copy()
    ca["HISP"] = np.select([ca["ETHNIC"].isin([1, 2, 3, 5]), ca["ETHNIC"] == 4], [1.0, 0.0], np.nan)
    ca["GROUP"] = ca["HISP"].map({1.0: "Hispanic", 0.0: "Non-Hispanic"})
    out = {"n_ca_outpatient": int(len(ca)), "hispanic_share": pct(ca["HISP"].mean())}

    reasons = {1: "completed", 2: "dropped_out", 3: "terminated", 4: "transferred", 5: "incarcerated", 6: "death", 7: "other"}
    tab = pd.crosstab(ca["GROUP"], ca["REASON"], normalize="index") * 100
    out["reason_by_group"] = {g: {reasons[k]: round(v, 1) for k, v in row.items()} for g, row in tab.iterrows()}

    h = ca[ca["HISP"] == 1]
    out["completion_by_origin"] = {ORIGIN[k]: {"n": int(len(g)), "completed": pct((g["REASON"] == 1).mean()),
                                               "dropped_out": pct((g["REASON"] == 2).mean())}
                                   for k, g in h.groupby("ETHNIC")}

    known = ca[ca["DAYWAIT"] >= 0]
    out["daywait_by_group"] = (pd.crosstab(known["GROUP"], known["DAYWAIT"], normalize="index") * 100).round(1).to_dict(orient="index")
    out["dropout_by_daywait"] = {int(k): {"n": int(len(g)), "dropped_out": pct((g["REASON"] == 2).mean())}
                                 for k, g in known.groupby("DAYWAIT")}

    co = ca[ca["PSYPROB"] == 1]
    out["co_occurring"] = {"n": int(len(co)), "share": pct((ca["PSYPROB"] == 1).mean()),
                           "completed_by_group": {g: pct((d["REASON"] == 1).mean()) for g, d in co.groupby("GROUP")}}

    out["adjusted_or_completed"] = adjusted_or(ca, lambda d: d["REASON"] == 1)
    out["adjusted_or_dropout"] = adjusted_or(ca, lambda d: d["REASON"] == 2)

    cb = ca[ca["CBSA2020"] > 0].groupby("CBSA2020").agg(n=("HISP", "size"), hisp=("HISP", "mean"))
    cb = cb[cb["n"] >= 500].sort_values("hisp")
    out["hispanic_share_by_metro"] = {"n_metros": int(len(cb)),
                                      "min": pct(cb["hisp"].min()), "median": pct(cb["hisp"].median()),
                                      "max": pct(cb["hisp"].max()),
                                      "metros": {str(int(k)): {"n": int(r.n), "hispanic_pct": pct(r.hisp)} for k, r in cb.iterrows()}}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "hispanic_share_by_metro"}, indent=1))
    m = out["hispanic_share_by_metro"]
    print(f"Hispanic share across {m['n_metros']} CA metros (500+ episodes): min {m['min']}%, median {m['median']}%, max {m['max']}%")


if __name__ == "__main__":
    main()
