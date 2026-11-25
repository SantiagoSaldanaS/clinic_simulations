"""
Model settings. Every value has a source or is marked as an assumption, and
RANGES gives the low/high values that sensitivity.py tests.

The treatment chain and the wait effect come from TEDS-D 2024 (California
outpatient), via estimate_from_tedsd.py -> outputs/params_tedsd_ca.json.
"""
import json
import os

import numpy as np
from scipy.optimize import brentq

DEFAULT_CHAIN_FILE = os.path.join(os.path.dirname(__file__), "outputs", "params_tedsd_ca.json")
SUMMARY_FILE = os.path.join(os.path.dirname(__file__), "outputs", "tedsd_ca_summary.json")


def _Q(h):
    K = len(h)
    Q = np.zeros((K, K))
    for s in range(K):
        if s < K - 1:
            Q[s, s + 1] = h[s, 3]
        Q[s, s] = 1 - h[s].sum()
    return Q


EARLY_WEEKS = 4  # the wait effect applies during the first month in care (NIATx tracked sessions 2-4)


def raise_dropout(row, mult):
    """Multiply this week's odds of dropping out by `mult`; scale the other moves to keep the row summing to 1."""
    row = np.array(row, float)
    full = np.append(row, max(0.0, 1 - row.sum()))  # complete, drop, other, advance, stay
    d = full[1]
    d_new = d * mult / (1 - d + d * mult)
    scale = (1 - d_new) / (1 - d) if d < 1 else 0.0
    full = full * scale
    full[1] = d_new
    return full


def early_outcome(h, log_or, wait_weeks):
    """Absorption probabilities and P(still in care after EARLY_WEEKS) for a patient who waited `wait_weeks`."""
    h = np.asarray(h, float)
    K = len(h)
    mult = np.exp(log_or * min(wait_weeks, 12))
    v = np.zeros(K); v[0] = 1
    absorbed = np.zeros(3)
    for _ in range(EARLY_WEEKS):
        nv = np.zeros(K)
        for s in range(K):
            if v[s] == 0:
                continue
            r = raise_dropout(h[s], mult)
            absorbed += v[s] * r[:3]
            if s < K - 1:
                nv[s + 1] += v[s] * r[3]
            nv[s] += v[s] * r[4]
        v = nv
    N = np.linalg.inv(np.eye(K) - _Q(h))
    return absorbed + v @ N @ h[:, :3], v.sum()


def stage1_log_or_from_episode(h, episode_log_or, weeks=2.0):
    """Early-weeks effect that reproduces the episode-level dropout odds ratio from TEDS-D, at a 2-week wait."""
    logit = lambda x: np.log(x / (1 - x))
    target = episode_log_or * weeks
    f = lambda g: logit(early_outcome(h, g, weeks)[0][1]) - logit(early_outcome(h, 0.0, weeks)[0][1]) - target
    return brentq(f, 0.0, 5.0) if target > 0 else 0.0


def stage1_log_or_from_niatx(h, ratio=40.8 / 68.2, wait_weeks=30 / 7):
    """Early-weeks effect that reproduces NIATx's drop in session-4 attendance after a 30-day wait."""
    base = early_outcome(h, 0.0, 0)[1]
    f = lambda g: early_outcome(h, g, wait_weeks)[1] / base - ratio
    return brentq(f, 0.0, 5.0) if f(5.0) < 0 else 5.0


def load_settings(chain_file=None):
    chain_file = chain_file or DEFAULT_CHAIN_FILE
    if not os.path.exists(chain_file):
        raise SystemExit(f"{chain_file} not found. Run: python estimate_from_tedsd.py <tedsd_puf_2024.csv> "
                         f"--out {DEFAULT_CHAIN_FILE}")
    fit = json.load(open(chain_file))
    chains = fit["stage_probs"]
    ep = fit.get("wait_effect", {}).get("log_or_per_week", 0.0)
    g_teds = stage1_log_or_from_episode(chains["Non-Hispanic"], max(ep, 0.0))
    g_niatx = stage1_log_or_from_niatx(chains["Non-Hispanic"])
    latino = fit.get("hispanic_share", 0.476)

    settings = {
        "stage_probs": chains,
        "latino_share": round(latino, 3),
        "spanish_pref_share": 0.27,
        "caseload_per_clinician": 30,
        "general_clinicians": 2,
        "bilingual_clinicians": 1,
        "offered_load": 0.95,
        "renege_prob": 0.21,
        "wait_log_or_per_week": round(g_teds, 4),
        "bilingual_spillover": True,
        "turnover_annual": 0.332,
        "vacancy_weeks_general": 8.7,
        "bilingual_vacancy_factor": 1.5,
    }
    metro = {}
    if os.path.exists(SUMMARY_FILE):
        metro = json.load(open(SUMMARY_FILE)).get("hispanic_share_by_metro", {})
    ranges = {
        "latino_share": (0.27, 0.65,
            f"TEDS-D 2024: Hispanic share of CA outpatient episodes is {latino:.1%} statewide; across 18 metros "
            f"with 500+ episodes it runs {metro.get('min', '?')}% to {metro.get('max', '?')}%. Range = Sacramento "
            "(26.7%) to Salinas (64.8%)."),
        "spanish_pref_share": (0.15, 0.39,
            "CA Latinos age 5+, ACS 2021 via CHCF Latino Health Almanac: 15% speak English 'not well' or "
            "'not at all'; 27% speak it less than 'very well' (Census definition, the base). High: San Benito "
            "County Behavioral Health, where 64% of SUD clients are Latino and about 1 in 4 clients prefers "
            "Spanish (25/64 = 39%)."),
        "caseload_per_clinician": (20, 40,
            "Outpatient SUD counselor survey: mean caseload 30, range 6-42 (Addict Sci Clin Pract 2021); "
            "New Jersey outpatient staffing standard 1:35."),
        "offered_load": (0.80, 1.10,
            "Assumption: referrals relative to capacity. Base 0.95 keeps the clinic near the 85% occupancy implied "
            "by N-SUMHSS (about 76 outpatient clients per CA facility vs 90 slots)."),
        "renege_prob": (0.10, 0.30,
            "Waitlist tolerance about one month, about 40% leave within two weeks (Kaplan & Johri 2000, as "
            "reported in PMC2766557). Weekly 1-exp(-7/30) = 0.21."),
        "wait_log_or_per_week": (0.0, round(g_niatx, 4),
            f"Base: TEDS-D 2024 CA logistic fit, episode dropout OR {np.exp(ep):.3f} per week waited, converted "
            f"to an effect on weekly dropout odds during the first {EARLY_WEEKS} weeks in care ({g_teds:.3f} per "
            f"week waited). High: NIATx session-4 attendance 68.2% at 0 days vs 40.8% at 30 days (Hoffman et al. "
            f"2011), which needs {g_niatx:.3f}."),
        "turnover_annual": (0.19, 0.50,
            "Counselor turnover 33.2% per year in 27 SUD treatment organizations (Eby et al. 2010); "
            "earlier estimates they cite run 19% to 50%."),
        "vacancy_weeks_general": (4.3, 13.0,
            "County behavioral health hiring often takes one to three months (Santa Cruz County Civil "
            "Grand Jury 2023). Base two months."),
        "bilingual_vacancy_factor": (1.0, 3.0,
            "Assumption: bilingual posts take longer to fill. No published figure found; base 1.5x."),
    }
    meta = {"chain_source": fit.get("note", chain_file), "ranges": ranges,
            "lr_test": fit.get("lr_test"), "wait_effect": fit.get("wait_effect"),
            "clinic_size_note": "N-SUMHSS 2023 California: 70,614 clients in outpatient SUD treatment across 933 "
                                "outpatient facilities, about 76 per facility. Three clinicians at caseload 30 = 90 slots."}
    return settings, meta


RANGES_KEYS = ["latino_share", "spanish_pref_share", "caseload_per_clinician", "offered_load",
               "renege_prob", "wait_log_or_per_week", "turnover_annual", "vacancy_weeks_general",
               "bilingual_vacancy_factor"]
