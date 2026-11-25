"""Static figures for slides, from outputs/results.json, outputs/sensitivity.json and the TEDS-D fit."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "outputs"
R = json.load(open(os.path.join(out, "results.json")))
FIT = json.load(open(os.path.join(out, "params_tedsd_ca.json")))
SENS_PATH = os.path.join(out, "sensitivity.json")
TYPES = ["Latino, Spanish-preferring", "Latino, English-proficient", "Non-Latino"]
# Blue/orange plus marker shape, so the pairs stay apart for colorblind readers.
COL = {"Hispanic": "#d95f02", "Non-Hispanic": "#1f5fa8", TYPES[0]: "#d95f02", TYPES[2]: "#1f5fa8"}
MARK = {TYPES[0]: "^", TYPES[2]: "s"}
TXT, TXT2, GRID = "#14161a", "#4b5058", "#e1e3e7"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TXT2, "xtick.color": TXT2,
                     "ytick.color": TXT2, "axes.titlecolor": TXT, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlelocation": "left", "axes.titlesize": 12})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=180)
    plt.close(fig)


# 1. Fit check: observed vs model-implied LOS x exit, per group
bands = ["1-7 d", "8-14", "15-30", "31-45", "46-60", "61-90", "91-120", "121-365"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, g in zip(axes, ["Hispanic", "Non-Hispanic"]):
    fc = FIT["fits"][g]["fit_check"]
    obs, exp_ = np.array(fc["observed"]), np.array(fc["expected"])
    n = obs.sum()
    x = np.arange(len(bands))
    for k, (name, ls) in enumerate(zip(["Completed", "Dropped out", "Other exit"], ["-", "--", ":"])):
        ax.bar(x + (k - 1) * 0.27, 100 * obs[:, k] / n, 0.25, color=["#3b3b38", "#8a877e", "#c9c5ba"][k], label=f"{name} (observed)")
        ax.plot(x + (k - 1) * 0.27, 100 * exp_[:, k] / n, "o", ms=4, color=COL[g], label="Model" if k == 0 else None)
    ax.set_xticks(x, bands, fontsize=9)
    ax.set_title(f"{g}: {FIT['fits'][g]['n']:,} episodes, K = {FIT['fits'][g]['K']} stages")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("% of episodes")
axes[1].legend(frameon=False, fontsize=8, loc="upper left")
fig.text(0.01, 0.01, "TEDS-D 2024, California outpatient. Bars: observed length of stay and exit type. Dots: fitted chain.",
         color=TXT2, fontsize=8)
save(fig, "fig1_chain_fit.png")

# 2. Chain outcomes with bootstrap intervals, plus the simulation check
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(3); w = 0.36
for i, g in enumerate(["Hispanic", "Non-Hispanic"]):
    f = FIT["fits"][g]
    b = 100 * np.array(f["absorption"])
    lo, hi = 100 * np.array(f["absorption_ci"][0]), 100 * np.array(f["absorption_ci"][1])
    xs = x + (i - 0.5) * (w + 0.03)
    ax.bar(xs, b, w, color=COL[g], label=g)
    ax.errorbar(xs, b, yerr=[b - lo, hi - b], fmt="none", ecolor=TXT, lw=1)
    sim = R["validation"][g]["simulated"]
    ax.scatter(xs, sim, s=36, color="white", edgecolor=TXT, zorder=3, label="Simulated" if i == 0 else None)
    for xi, v in zip(xs, b):
        ax.text(xi, v + 2.2, f"{v:.1f}%", ha="center", fontsize=9, color=TXT)
ax.set_xticks(x, ["Completed", "Dropped out", "Other exit"])
ax.set_ylabel("% of episodes"); ax.set_ylim(0, 60)
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper right")
ax.set_title("How outpatient episodes end, absorbing chain B = (I - Q)⁻¹R, 95% bootstrap CI")
save(fig, "fig2_chain_outcomes.png")

# 3. Staffing by community: completion and giving up
rows = [r for r in R["staffing"] if r["bilingual"] > 0]
comms = list(dict.fromkeys(r["community"] for r in rows))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
for ax, key, ylab, ylim in [(a1, "pct_completed", "% of referrals\nwho complete", (15, 30)),
                            (a2, "pct_reneged", "% of referrals who\ngive up waiting", (0, 18))]:
    for ci, c in enumerate(comms):
        for r in rows:
            if r["community"] != c:
                continue
            xo = ci + (r["bilingual"] - 1.5) * 0.36
            for t in (TYPES[0], TYPES[2]):
                m, h = r[f"{t}|{key}"]
                ax.errorbar(xo + (0.07 if t == TYPES[2] else -0.07), m, yerr=h, fmt=MARK[t], ms=7, color=COL[t], lw=1.5)
            if ax is a2:
                ax.text(xo, ylim[0] + 0.6, f"{r['bilingual']}B", ha="center", fontsize=8, color=TXT2)
    ax.set_ylim(*ylim); ax.set_ylabel(ylab)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
a2.set_xticks(range(len(comms)), [c.replace(" profile", "").replace(" metro", "") for c in comms], fontsize=9)
for t in (TYPES[0], TYPES[2]):
    a1.plot([], [], MARK[t], color=COL[t], label=t)
a1.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
a1.set_title("Three-clinician clinic with 1 or 2 bilingual clinicians (B), communities ordered by Latino share, 95% CI")
save(fig, "fig3_staffing_by_community.png")

# 4. Waitlist and vacancies over time, San Benito profile
from settings import load_settings
from sim import simulate
p0, _ = load_settings()
case = dict(p0, latino_share=0.64, spanish_pref_share=0.39)
for seed in range(11, 200):  # first run with a bilingual vacancy in the middle of the window, so the plot shows one
    tr = simulate(case, seed=seed, record=True)["log"]
    if (tr[100:250, 6] == 0).any():
        break
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
a1.plot(tr[:, 0], tr[:, 1], color=COL[TYPES[0]], lw=1.6, label="Waiting for a bilingual clinician")
a1.plot(tr[:, 0], tr[:, 2], color=COL[TYPES[2]], lw=1.6, ls="--", label="Waiting for any clinician")
a1.set_ylabel("People waiting"); a1.legend(frameon=False, fontsize=9, loc="upper left")
a1.yaxis.grid(True, color=GRID); a1.set_axisbelow(True)
a1.set_title(f"One run (seed {seed}, picked because its bilingual post falls vacant), San Benito profile, 1 bilingual of 3")
a2.step(tr[:, 0], tr[:, 6], where="post", color=COL[TYPES[0]], lw=1.6, label="Bilingual")
a2.step(tr[:, 0], tr[:, 5], where="post", color=COL[TYPES[2]], lw=1.6, ls="--", label="English only")
a2.set_yticks([0, 1, 2]); a2.set_ylabel("Clinicians\non staff", fontsize=9); a2.set_xlabel("Week")
a2.legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
save(fig, "fig4_waitlist_trace.png")

# 5. Tornado
if os.path.exists(SENS_PATH):
    S = json.load(open(SENS_PATH))
    base = S["reference"]["gave_up"][0]
    labels = {"latino_share": "Latino share of referrals", "spanish_pref_share": "Spanish-preferring share of Latinos",
              "caseload_per_clinician": "Caseload per clinician", "offered_load": "Demand vs capacity",
              "renege_prob": "Weekly prob. of giving up", "wait_log_or_per_week": "Wait effect on dropout",
              "turnover_annual": "Annual staff turnover", "vacancy_weeks_general": "Weeks to fill a post",
              "bilingual_vacancy_factor": "Extra time to fill a bilingual post"}
    items = sorted(S["oneway"].items(), key=lambda kv: abs(kv[1]["high"]["gave_up"][0] - kv[1]["low"]["gave_up"][0]))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (k, v) in enumerate(items):
        lo, hi = v["low"]["gave_up"][0], v["high"]["gave_up"][0]
        ax.plot([lo, hi], [i, i], color=GRID, lw=6, solid_capstyle="butt", zorder=1)
        ax.scatter([lo], [i], marker="o", s=40, color="#57564f", zorder=2)
        ax.scatter([hi], [i], marker="D", s=40, color="#d95f02", zorder=2)
        ax.text(max(lo, hi) + 0.4, i, f"{v['low']['value']:g} to {v['high']['value']:g}", va="center", fontsize=8, color=TXT2)
    ax.axvline(base, color=TXT, lw=1)
    ax.set_yticks(range(len(items)), [labels[k] for k, _ in items], fontsize=9)
    ax.set_xlabel("% of Spanish-preferring referrals who give up waiting")
    ax.scatter([], [], marker="o", color="#57564f", label="Low value"); ax.scatter([], [], marker="D", color="#d95f02", label="High value")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("One-way sensitivity, San Benito profile with 1 bilingual of 3")
    save(fig, "fig5_tornado.png")
print("figures saved")
