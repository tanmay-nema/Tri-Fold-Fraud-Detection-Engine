"""
Layer 2 - Statistical Anomaly Detection (3-SD check + Isolation Forest)
Two complementary checks, a claim is flagged if EITHER fires:
  (a) 3-SD check:  is bill or stay more than 3 standard deviations away from typical?
                   (simple, explainable, catches a single wildly wrong number)
  (b) Isolation Forest: catches unusual COMBINATIONS that no single number reveals.
Isolation Forest idea: learn what a "normal" claim looks like from clean history, then flag claims
that look very different. Think of it as: "how easy is it to separate this claim
from all the normal ones?"  Odd claims are easy to isolate -> low score -> flagged.

To compare fairly across packages we do not use raw rupees/days. We use RATIOS:
    bill_ratio = bill / typical bill of that package      (1.0 = typical)
    stay_ratio = days  / typical days of that package     (1.0 = typical)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def _stay(df):
    return (pd.to_datetime(df["discharge_date"]) - pd.to_datetime(df["admit_date"])).dt.days


def _features(df, medians):
    out = pd.DataFrame(index=df.index)
    med_bill = df["billing_code"].map(medians["bill"])
    med_stay = df["billing_code"].map(medians["stay"])
    out["bill_ratio"] = df["bill_amount"] / med_bill
    out["stay_ratio"] = _stay(df) / med_stay
    return out


def reference_features(baseline):
    """Medians per package + feature table of the clean baseline."""
    b = baseline.copy()
    b["_stay"] = _stay(b)
    medians = {"bill": b.groupby("billing_code")["bill_amount"].median().to_dict(),
               "stay": b.groupby("billing_code")["_stay"].median().to_dict()}
    return medians, _features(b, medians)


def run(df, baseline, contamination=0.05):
    df = df.copy()
    medians, ref = reference_features(baseline)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(ref)                                   # learn "normal" from clean history only
    spread = ref.std()                               # normal spread of each ratio in clean history

    feats = _features(df, medians)
    scorable = feats.notna().all(axis=1)             # component rows (CMP_*) have no package median
    df["bill_ratio"] = feats["bill_ratio"].round(2)
    df["stay_ratio"] = feats["stay_ratio"].round(2)
    df["l2_zmax"] = None
    df["l2_score"] = None
    df["l2_flag"] = 0
    df["l2_reason"] = ""
    if scorable.any():
        X = feats[scorable]
        df.loc[scorable, "l2_score"] = model.decision_function(X).round(3)   # lower = more unusual
        z = ((X - 1) / spread).abs()                 # distance from "typical" in standard deviations
        df.loc[scorable, "l2_zmax"] = z.max(axis=1).round(1)
        by_sd = (z > 3).any(axis=1)
        by_forest = pd.Series(model.predict(X) == -1, index=X.index)
        df.loc[scorable, "l2_flag"] = (by_sd | by_forest).astype(int)
    for i in df.index[df["l2_flag"] == 1]:
        parts = []
        if abs(df.at[i, "bill_ratio"] - 1) > 0.5:
            parts.append(f"bill is {df.at[i, 'bill_ratio']}x package median")
        if abs(df.at[i, "stay_ratio"] - 1) > 0.5:
            parts.append(f"stay is {df.at[i, 'stay_ratio']}x package median")
        df.at[i, "l2_reason"] = f"L2 outlier ({df.at[i, 'l2_zmax']} SD): " + ("; ".join(parts) or "unusual combination")
    return df, ref


def plot(df, ref, path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(ref["bill_ratio"], ref["stay_ratio"], c="lightgrey", s=35, label="Clean history (what the model learned)")
    ok = df[(df["l2_flag"] == 0) & df["l2_score"].notna()]
    bad = df[df["l2_flag"] == 1]
    ax.scatter(ok["bill_ratio"], ok["stay_ratio"], c="tab:blue", s=70, label="Input claim - normal")
    ax.scatter(bad["bill_ratio"], bad["stay_ratio"], c="tab:red", s=110, marker="X", label="Input claim - flagged by Layer 2")
    for _, r in bad.iterrows():
        ax.annotate(r["claim_id"], (r["bill_ratio"], r["stay_ratio"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Bill ratio (1.0 = typical for that package)")
    ax.set_ylabel("Stay ratio (1.0 = typical for that package)")
    ax.set_title("Layer 2: Isolation Forest - normal vs unusual claims")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
