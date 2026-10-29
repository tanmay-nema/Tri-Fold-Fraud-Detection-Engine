"""
app.py - runs the whole Tri-Fold experiment:  python app.py
Flow:  input CSV -> copy -> Layer 1 -> Layer 2 -> Layer 3 -> output CSV + graphs
"""
import os
import shutil
import pandas as pd
import layer1, layer2, layer3

os.makedirs("output", exist_ok=True)
if not os.path.exists("data/input_claims.csv"):
    import generate_data  # noqa: F401  (creates the data files)

# 0. copy the input so the original is never touched
shutil.copy("data/input_claims.csv", "output/working_copy.csv")
df = pd.read_csv("output/working_copy.csv")
baseline = pd.read_csv("data/baseline_claims.csv")
print(f"Loaded {len(df)} input claims, {len(baseline)} clean baseline claims\n")

# 1. Layer 1 - hard rules
df = layer1.run(df)
print(f"Layer 1 (rules)         flagged: {df['l1_flag'].sum()} claims")

# 2. Layer 2 - Isolation Forest
df, ref = layer2.run(df, baseline)
print(f"Layer 2 (3-SD + IForest)  flagged: {df['l2_flag'].sum()} claims")
layer2.plot(df, ref, "output/graph_layer2_anomaly.png")

# 3. Layer 3 - graph collision
df, G, suspects, colliding = layer3.run(df, baseline)
print(f"Layer 3 (graph)         flagged: {df['l3_flag'].sum()} claims "
      f"(suspects: {suspects}, collisions: {colliding})")
layer3.plot(G, suspects, colliding, "output/graph_layer3_network.png")

# 4. combine + save
df["any_flag"] = ((df["l1_flag"] + df["l2_flag"] + df["l3_flag"]) > 0).astype(int)
df["detected_layers"] = df.apply(
    lambda r: "+".join(f"L{n}" for n in (1, 2, 3) if r[f"l{n}_flag"]) or "none", axis=1)

# 5. verify against the answer key (used ONLY here, never by the layers)
truth = pd.read_csv("data/ground_truth.csv")
df = df.merge(truth, on="claim_id", how="left")
df["match"] = (df["detected_layers"] == df["expected_layers"]).map({True: "OK", False: "MISMATCH"})
df.to_csv("output/output_claims.csv", index=False)

print("\n" + df[["claim_id", "billing_code", "detected_layers", "expected_layers", "match"]].to_string(index=False))
print(f"\nMatched answer key: {(df['match'] == 'OK').sum()}/{len(df)}")
print("Saved: output/output_claims.csv, output/graph_layer2_anomaly.png, output/graph_layer3_network.png")
