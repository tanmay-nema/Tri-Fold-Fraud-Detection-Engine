"""
Layer 3 - Graph Collision Analysis (NetworkX)
Idea: draw a map. Every doctor and every hospital is a dot; a line joins a doctor
to each hospital where they billed a claim.
  Step A (graph):  a doctor connected to 2+ hospitals is a SUSPECT.
  Step B (time):   a suspect is FLAGGED only if two of their procedures at different
                   hospitals overlap in time (one person cannot operate in two places at once).
A doctor who works at two hospitals on different days stays clean.
"""
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(df, context):
    """df = input claims, context = background claims (used to draw a fuller graph)."""
    df = df.copy()
    allc = pd.concat([context.assign(src="context"), df.assign(src="input")], ignore_index=True)
    allc["_s"] = pd.to_datetime(allc["procedure_start"])
    allc["_e"] = pd.to_datetime(allc["procedure_end"])

    G = nx.Graph()                                    # Step A: build the doctor-hospital graph
    for _, r in allc.iterrows():
        G.add_node(r["doctor_reg_no"], kind="doctor")
        G.add_node(r["hospital_id"], kind="hospital")
        G.add_edge(r["doctor_reg_no"], r["hospital_id"])
    suspects = [n for n, d in G.nodes(data=True)
                if d["kind"] == "doctor" and G.degree(n) >= 2]

    reasons, colliding = {}, set()                    # Step B: check time overlap for suspects only
    for doc in suspects:
        g = allc[allc["doctor_reg_no"] == doc]
        idx = list(g.index)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                x, y = g.loc[idx[a]], g.loc[idx[b]]
                if x["hospital_id"] != y["hospital_id"] and x["_s"] < y["_e"] and y["_s"] < x["_e"]:
                    colliding.add(doc)
                    for me, other in ((x, y), (y, x)):
                        if me["src"] == "input":
                            reasons[me["claim_id"]] = (f"L3 collision: {doc} billed at {me['hospital_id']} "
                                                       f"({me['procedure_start'][-5:]}-{me['procedure_end'][-5:]}) and at "
                                                       f"{other['hospital_id']} ({other['procedure_start'][-5:]}-{other['procedure_end'][-5:]}) "
                                                       f"on the same day ({other['claim_id']})")
    df["l3_flag"] = df["claim_id"].isin(reasons).astype(int)
    df["l3_reason"] = df["claim_id"].map(reasons).fillna("")
    return df, G, suspects, sorted(colliding)


def plot(G, suspects, colliding, path):
    pos = nx.spring_layout(G, seed=7, k=0.6)
    hosp = [n for n, d in G.nodes(data=True) if d["kind"] == "hospital"]
    docs_ok = [n for n, d in G.nodes(data=True) if d["kind"] == "doctor" and n not in suspects]
    docs_susp = [n for n in suspects if n not in colliding]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    bad_edges = [(u, v) for u, v in G.edges if u in colliding or v in colliding]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="lightgrey")
    nx.draw_networkx_edges(G, pos, edgelist=bad_edges, ax=ax, edge_color="tab:red", width=2.5)
    nx.draw_networkx_nodes(G, pos, nodelist=docs_ok, node_size=120, node_color="tab:green", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=docs_susp, node_size=200, node_color="orange", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=colliding, node_size=300, node_color="tab:red", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=hosp, node_size=900, node_color="tab:blue", node_shape="s", ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in hosp + suspects}, font_size=8,
                            font_color="black", ax=ax)
    handles = [plt.Line2D([], [], marker="s", ls="", color="tab:blue", ms=10, label="Hospital"),
               plt.Line2D([], [], marker="o", ls="", color="tab:green", ms=8, label="Doctor (one hospital)"),
               plt.Line2D([], [], marker="o", ls="", color="orange", ms=8, label="Doctor at 2+ hospitals, no time clash (cleared)"),
               plt.Line2D([], [], marker="o", ls="", color="tab:red", ms=10, label="Doctor with time collision (FLAGGED)")]
    ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.set_title("Layer 3: Doctor-Hospital graph and collision detection")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
