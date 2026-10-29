"""
Layer 1 - Rule-Based Hard Gates
Idea: simple IF-THEN rules written by humans. No learning, no maths.
Catches claims that are impossible no matter what the numbers are.
"""
import pandas as pd

# ---- the rule tables (in the real prototype these come from IRDAI / package master data) ----
ALLOWED_SEX = {"PKG_HYST": "F"}                      # hysterectomy only for female patients
PACKAGE_COMPONENTS = {                                # items already INCLUDED inside a package
    "PKG_APPY": ["CMP_ANAES", "CMP_OT", "CMP_ROOM"],
    "PKG_CAT":  ["CMP_ANAES", "CMP_LENS"],
    "PKG_HYST": ["CMP_ANAES", "CMP_OT", "CMP_ROOM"],
    "PKG_KNEE": ["CMP_ANAES", "CMP_OT", "CMP_IMPLANT"],
}


def run(df):
    df = df.copy()
    admit = pd.to_datetime(df["admit_date"])
    discharge = pd.to_datetime(df["discharge_date"])
    reasons = {i: [] for i in df.index}
    is_package = df["billing_code"].str.startswith("PKG_")

    # Rule 1: impossible biological procedure (sex vs procedure)
    for i, r in df.iterrows():
        need = ALLOWED_SEX.get(r["billing_code"])
        if need and r["sex"] != need:
            reasons[i].append(f"R1 impossible procedure: {r['billing_code']} billed for sex={r['sex']}")

    # Rule 2: unbundled billing (a component billed separately although its package is also billed)
    for i, r in df[~is_package].iterrows():
        pkgs = df[is_package & (df["patient_id"] == r["patient_id"])]
        for j, p in pkgs.iterrows():
            inside = r["billing_code"] in PACKAGE_COMPONENTS.get(p["billing_code"], [])
            same_stay = admit[j] <= admit[i] <= discharge[j]
            if inside and same_stay:
                reasons[i].append(f"R2 unbundled: {r['billing_code']} is already inside {p['billing_code']} ({p['claim_id']})")

    # Rule 3: overlapping hospital admissions for the same patient (package claims only)
    pk = df[is_package]
    for pid, g in pk.groupby("patient_id"):
        ids = list(g.index)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                if admit[i] < discharge[j] and admit[j] < discharge[i]:
                    reasons[i].append(f"R3 overlapping admission with {df.at[j, 'claim_id']} ({df.at[j, 'hospital_id']})")
                    reasons[j].append(f"R3 overlapping admission with {df.at[i, 'claim_id']} ({df.at[i, 'hospital_id']})")

    df["l1_flag"] = [1 if reasons[i] else 0 for i in df.index]
    df["l1_reason"] = ["; ".join(reasons[i]) for i in df.index]
    return df
