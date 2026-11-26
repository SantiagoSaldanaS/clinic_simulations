"""Write animation/params.json: base settings plus per-metro profiles from outputs/communities.json."""
import json
import os

from settings import load_settings

here = os.path.dirname(os.path.abspath(__file__))
p, meta = load_settings()
base = {k: p[k] for k in ("caseload_per_clinician", "renege_prob", "wait_log_or_per_week", "bilingual_spillover",
                          "turnover_annual", "vacancy_weeks_general", "bilingual_vacancy_factor")}
communities = json.load(open(os.path.join(here, "outputs", "communities.json")))
state = next(c for c in communities if c["name"] == "California")
# San Benito County has no metro of its own in TEDS-D; its county report gives the Latino and Spanish-preferring shares
communities.append({
    "name": "San Benito County", "n": 0, "latino_share": 0.64, "spanish_pref_share": 0.39,
    "stage_probs": state["stage_probs"], "chain_source": {"Hispanic": "statewide", "Non-Hispanic": "statewide"},
    "offered_load": state["offered_load"], "wait_8plus_days": state["wait_8plus_days"],
    "note": "The county's behavioral health plan reports that 64% of substance use clients are Latino and that about "
            "1 in 4 clients prefers Spanish. The treatment chains and the referral volume use the statewide fit.",
})
fit = json.load(open(os.path.join(here, "outputs", "params_tedsd_ca.json")))
summ = json.load(open(os.path.join(here, "outputs", "tedsd_ca_summary.json")))
sens = json.load(open(os.path.join(here, "outputs", "sensitivity.json")))
res = json.load(open(os.path.join(here, "outputs", "results.json")))
theory = {
    "fits": {g: {k: fit["fits"][g][k] for k in ("K", "n", "stage_probs", "absorption", "absorption_ci", "mean_weeks",
                                                 "mean_weeks_ci", "bic", "observed_shares")} for g in fit["fits"]},
    "lr_test": fit["lr_test"], "wait_effect": {k: fit["wait_effect"][k] for k in ("n", "or_per_week", "or_ci")},
    "wait_stage1": p["wait_log_or_per_week"], "wait_niatx": meta["ranges"]["wait_log_or_per_week"][1],
    "adjusted": {"completed": summ["adjusted_or_completed"], "dropout": summ["adjusted_or_dropout"]},
    "daywait_by_group": summ["daywait_by_group"], "n_ca": summ["n_ca_outpatient"],
    "prcc_gave_up": sens["global"]["prcc_gave_up"], "gave_up_quantiles": sens["global"]["gave_up_quantiles"],
    "reference_gave_up": sens["reference"]["gave_up"], "validation": res["validation"],
}
for k in ("adjusted",):
    for v in theory[k].values():
        v.pop("covariates", None)
out = {"base": base, "communities": communities, "theory": theory, "source": "TEDS-D 2024, California outpatient; see README"}
with open(os.path.join(here, "animation", "params.json"), "w") as f:
    json.dump(out, f)
print(f"Wrote animation/params.json with {len(communities)} communities")
