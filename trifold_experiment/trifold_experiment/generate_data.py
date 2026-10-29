"""
generate_data.py  -  builds the experiment data (run once).

Creates 3 files in data/:
  input_claims.csv     10 showcase claims, some with planted flaws
  baseline_claims.csv  ~150 clean synthetic claims (Layer 2 learns "normal" from these,
                       Layer 3 uses them as background for the graph)
  ground_truth.csv     answer key: which layer SHOULD flag each showcase claim
                       (the layers never read this file; app.py only uses it to verify)
"""
import random
import pandas as pd

random.seed(42)

# package -> (median bill in INR, median stay in days)
PACKAGES = {
    "PKG_APPY": (45000, 3),    # appendectomy
    "PKG_CAT":  (28000, 2),    # cataract surgery
    "PKG_HYST": (70000, 4),    # hysterectomy
    "PKG_KNEE": (180000, 5),   # knee replacement
}
HOSPITALS = ["H01", "H02", "H03", "H04", "H05"]
# every hospital has 3 doctors, e.g. H02 -> DR201, DR202, DR203
DOCTORS = {h: [f"DR{h[1:]}{k}" for k in (1, 2, 3)] for h in HOSPITALS}
COLS = ["claim_id", "patient_id", "sex", "age", "billing_code", "admit_date", "discharge_date",
        "bill_amount", "doctor_reg_no", "hospital_id", "procedure_start", "procedure_end"]


def make_row(cid, pid, sex, age, code, admit, stay, bill, doc, hosp, start_h, dur_h):
    admit = pd.Timestamp(admit)
    start = admit + pd.Timedelta(hours=start_h)
    return dict(claim_id=cid, patient_id=pid, sex=sex, age=age, billing_code=code,
                admit_date=admit.date(), discharge_date=(admit + pd.Timedelta(days=stay)).date(),
                bill_amount=int(bill), doctor_reg_no=doc, hospital_id=hosp,
                procedure_start=start.strftime("%Y-%m-%d %H:%M"),
                procedure_end=(start + pd.Timedelta(hours=dur_h)).strftime("%Y-%m-%d %H:%M"))


# ---------- baseline: clean, realistic-looking claims ----------
base = []
for i in range(150):
    code = random.choice(list(PACKAGES))
    med_bill, med_stay = PACKAGES[code]
    sex = "F" if code == "PKG_HYST" else random.choice(["M", "F"])
    age = random.randint(30, 60) if code == "PKG_HYST" else random.randint(25, 80)
    hosp = random.choice(HOSPITALS)
    doc = random.choice(DOCTORS[hosp])          # each doctor stays at ONE hospital
    admit = pd.Timestamp("2026-01-05") + pd.Timedelta(days=random.randint(0, 50))
    base.append(make_row(f"B{i+1:03d}", f"P{1001+i}", sex, age, code, admit,
                         max(1, med_stay + random.choice([-1, 0, 0, 0, 1])),
                         med_bill * random.uniform(0.9, 1.1), doc, hosp,
                         random.randint(8, 14), random.choice([1, 2, 3])))

# a doctor who LEGITIMATELY works at two hospitals (different days) -> graph must NOT flag her
for j, (hosp, day) in enumerate([("H02", 10), ("H05", 14), ("H02", 20), ("H05", 24)]):
    base.append(make_row(f"B{151+j}", f"P{1151+j}", "M", 40 + j, "PKG_APPY",
                         pd.Timestamp("2026-03-01") + pd.Timedelta(days=day), 3, 45000 * 1.02,
                         "DR150", hosp, 10, 2))
pd.DataFrame(base, columns=COLS).to_csv("data/baseline_claims.csv", index=False)

# ---------- input: the 10 showcase claims ----------
rows = [
    # C001, C002, C004: clean claims
    make_row("C001", "P001", "M", 34, "PKG_APPY", "2026-02-10", 3, 44500, "DR101", "H01", 9, 2),
    make_row("C002", "P002", "F", 62, "PKG_CAT",  "2026-02-12", 2, 27500, "DR202", "H02", 9, 1),
    # C003: FLAW L1 + L2  -> hysterectomy on a MALE patient, bill ~5x normal
    make_row("C003", "P003", "M", 45, "PKG_HYST", "2026-02-14", 4, 350000, "DR102", "H01", 9, 3),
    # C004 + C005: FLAW L1 (unbundling) -> anaesthesia billed separately although inside the package
    make_row("C004", "P004", "F", 29, "PKG_APPY", "2026-02-16", 3, 46000, "DR303", "H03", 9, 2),
    make_row("C005", "P004", "F", 29, "CMP_ANAES", "2026-02-16", 3, 4500, "DR303", "H03", 9, 2),
    # C006 + C007: FLAW L1 (overlap) -> same patient admitted in two hospitals at the same time
    make_row("C006", "P006", "M", 58, "PKG_KNEE", "2026-03-01", 5, 181000, "DR201", "H02", 9, 3),
    make_row("C007", "P006", "M", 58, "PKG_APPY", "2026-03-05", 3, 45500, "DR301", "H03", 9, 2),
    # C008: FLAW L2 -> cataract with a huge bill and a 7-day stay
    make_row("C008", "P008", "F", 66, "PKG_CAT", "2026-04-02", 7, 150000, "DR501", "H05", 9, 1),
    # C009 + C010: FLAW L3 -> same doctor, two hospitals, overlapping procedure times
    make_row("C009", "P009", "M", 51, "PKG_APPY", "2026-04-10", 3, 45200, "DR999", "H01", 9, 2),
    make_row("C010", "P010", "F", 47, "PKG_APPY", "2026-04-10", 3, 44800, "DR999", "H04", 10, 2),
]
pd.DataFrame(rows, columns=COLS).to_csv("data/input_claims.csv", index=False)

truth = {"C001": "none", "C002": "none", "C003": "L1+L2", "C004": "none", "C005": "L1",
         "C006": "L1", "C007": "L1", "C008": "L2", "C009": "L3", "C010": "L3"}
pd.DataFrame({"claim_id": truth.keys(), "expected_layers": truth.values()}).to_csv(
    "data/ground_truth.csv", index=False)
print("Created data/input_claims.csv, data/baseline_claims.csv, data/ground_truth.csv")
