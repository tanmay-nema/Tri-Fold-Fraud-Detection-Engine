# Tri-Fold Fraud Engine - Small Experiment

Run:  pip install pandas scikit-learn networkx matplotlib
      python app.py            (data is auto-generated if missing; python generate_data.py regenerates it)

Flow: data/input_claims.csv -> output/working_copy.csv -> Layer 1 -> Layer 2 -> Layer 3 -> output/output_claims.csv + 2 graphs

| Layer | Method | Reads | Writes |
|---|---|---|---|
| layer1.py | 3 IF-THEN rules (sex vs procedure, unbundling, overlapping admissions) | claim rows | l1_flag, l1_reason |
| layer2.py | 3-SD check + Isolation Forest on bill/stay ratios, learned from data/baseline_claims.csv | bill, stay, package | bill_ratio, stay_ratio, l2_zmax, l2_score, l2_flag, l2_reason |
| layer3.py | Doctor-hospital graph (NetworkX) + procedure-time overlap check | doctor, hospital, procedure times | l3_flag, l3_reason |

Each layer reads only the original columns, never another layer's flags, so layers can be built and tested separately.
data/ground_truth.csv is an answer key used only by app.py at the very end to verify results.
