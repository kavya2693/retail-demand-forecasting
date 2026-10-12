"""
Feature builder for the graph-augmented replenishment model.

Produces a leakage-safe store x SKU x day feature matrix with the target
= forward H-day demand (units over [t+1 .. t+H]) — the lead-time quantity a
replenishment order must cover.

Feature blocks (flagged so train.py can ablate the graph):
  TEMPORAL   lags, rolling stats, calendar, price/promo, item age
  STATIC     product + store attributes
  GRAPH      SVD embeddings (product & store), substitution centrality,
             substitute-on-promo pressure, substitute & store-peer lagged demand

No PyTorch Geometric / Neo4j needed here — embeddings are computed with a
truncated SVD of the weighted adjacency (spectral embedding, deterministic).
The same graph in Neo4j/GDS (node2vec/FastRP) is a drop-in replacement.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import config as C

H = 7  # forecast horizon = replenishment lead-time window (days)

D = C.DATA_DIR


def _svd_embedding(edges: pd.DataFrame, a: str, b: str, w: str, nodes, dim: int, prefix: str):
    """Symmetric weighted adjacency -> truncated SVD node embedding."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), np.float32)
    for _, r in edges.iterrows():
        if r[a] in idx and r[b] in idx:
            i, j = idx[r[a]], idx[r[b]]
            A[i, j] = A[j, i] = r[w]
    # symmetric normalized adjacency
    d = A.sum(1); d[d == 0] = 1
    An = A / np.sqrt(np.outer(d, d))
    dim = min(dim, n - 1)
    U, S, _ = np.linalg.svd(An)
    emb = U[:, :dim] * np.sqrt(S[:dim])
    cols = [f"{prefix}{k}" for k in range(dim)]
    return pd.DataFrame(emb, columns=cols).assign(_node=nodes), cols


def build_features():
    sales = pd.read_csv(D / "sales.csv", parse_dates=["date"])
    cal   = pd.read_csv(D / "calendar.csv", parse_dates=["date"])
    prod  = pd.read_csv(D / "products.csv", parse_dates=["launch_date"])
    store = pd.read_csv(D / "stores.csv")
    e_sub = pd.read_csv(D / "edges_substitute.csv")
    e_sim = pd.read_csv(D / "edges_store_similar.csv")

    df = sales.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    g = df.groupby(["store_id", "sku_id"], sort=False)["units_sold"]

    # ---- TEMPORAL: lags + rolling (all shifted 1 day -> use info up to t) ----
    for L in (1, 7, 14, 28):
        df[f"lag_{L}"] = g.shift(L)
    s1 = g.shift(1)
    for W in (7, 28):
        df[f"rmean_{W}"] = s1.groupby([df.store_id, df.sku_id]).transform(lambda x: x.rolling(W, min_periods=1).mean())
        df[f"rstd_{W}"]  = s1.groupby([df.store_id, df.sku_id]).transform(lambda x: x.rolling(W, min_periods=2).std())
    df["promo_roll7"] = df.groupby(["store_id","sku_id"])["promo_flag"].transform(lambda x: x.shift(1).rolling(7,min_periods=1).mean())
    df["oos_roll7"]   = df.groupby(["store_id","sku_id"])["oos_flag"].transform(lambda x: x.shift(1).rolling(7,min_periods=1).mean())

    # ---- calendar + static merges ----
    df = df.merge(cal[["date","dow","month","is_holiday","season","temp_z"]], on="date", how="left")
    df = df.merge(prod, on="sku_id", how="left")
    df = df.merge(store, on="store_id", how="left")
    df["item_age"] = (df["date"] - df["launch_date"]).dt.days.clip(lower=0)
    df["gross_margin"] = (df["base_price"] - df["unit_cost"]) / df["base_price"]

    # ---- GRAPH: SVD embeddings ----
    prod_nodes = prod["sku_id"].tolist()
    # combine substitute (+complement optional) for product embedding
    emb_p, PCOLS = _svd_embedding(e_sub, "sku_a", "sku_b", "strength", prod_nodes, 16, "pemb")
    df = df.merge(emb_p.rename(columns={"_node":"sku_id"}), on="sku_id", how="left")
    store_nodes = store["store_id"].tolist()
    emb_s, SCOLS = _svd_embedding(e_sim, "store_a", "store_b", "similarity", store_nodes, 4, "semb")
    df = df.merge(emb_s.rename(columns={"_node":"store_id"}), on="store_id", how="left")

    # ---- GRAPH: substitution centrality (cold-start proxy) ----
    G = nx.from_pandas_edgelist(e_sub, "sku_a", "sku_b", ["strength"])
    pr = nx.pagerank(G, weight="strength") if G.number_of_edges() else {}
    deg = dict(G.degree())
    df["sub_pagerank"] = df["sku_id"].map(pr).fillna(0.0)
    df["sub_degree"]   = df["sku_id"].map(deg).fillna(0).astype(float)

    # ---- GRAPH: substitute promo/stockout pressure & neighbor lagged demand (same store/day) ----
    base = df[["store_id","sku_id","date","promo_flag","lag_7","oos_roll7"]].copy()
    m = e_sub.merge(base.rename(columns={"sku_id":"sku_b","promo_flag":"nb_promo",
                                         "lag_7":"nb_lag7","oos_roll7":"nb_oos"}),
                    on="sku_b", how="inner")
    m["wpromo"] = m["nb_promo"] * m["strength"]
    m["wdem"]   = m["nb_lag7"] * m["strength"]
    m["woos"]   = m["nb_oos"]  * m["strength"]
    agg = m.groupby(["store_id","sku_a","date"]).agg(
        subs_promo_pressure=("wpromo","sum"),
        subs_neighbor_demand=("wdem","mean"),
        subs_stockout_pressure=("woos","sum")).reset_index().rename(columns={"sku_a":"sku_id"})
    df = df.merge(agg, on=["store_id","sku_id","date"], how="left")

    # ---- GRAPH: store-peer demand (same SKU at SIMILAR_TO stores, lagged) ----
    peer = e_sim.merge(base.rename(columns={"store_id":"store_b","lag_7":"peer_lag7"}),
                       on=None, left_on="store_b", right_on="store_b", how="inner")
    peerside = base.rename(columns={"store_id":"store_b","lag_7":"peer_lag7"})[["store_b","sku_id","date","peer_lag7"]]
    peer = e_sim.merge(peerside, on="store_b", how="inner")
    peer["wpeer"] = peer["peer_lag7"] * peer["similarity"]
    pagg = peer.groupby(["store_a","sku_id","date"]).agg(store_peer_demand=("wpeer","mean")).reset_index().rename(columns={"store_a":"store_id"})
    df = df.merge(pagg, on=["store_id","sku_id","date"], how="left")

    for c in ["subs_promo_pressure","subs_neighbor_demand","subs_stockout_pressure","store_peer_demand"]:
        df[c] = df[c].fillna(0.0)

    # ---- TARGET: forward H-day demand ----
    fwd = df.groupby(["store_id","sku_id"])["units_sold"].transform(
        lambda x: x.shift(-1).rolling(H, min_periods=H).sum())
    df["y"] = fwd

    # ---- categoricals ----
    for c in ["category","subcategory","brand","vendor","format","region","season"]:
        df[c] = df[c].astype("category")

    # feature lists
    TEMPORAL = ["lag_1","lag_7","lag_14","lag_28","rmean_7","rmean_28","rstd_7","rstd_28",
                "promo_roll7","oos_roll7","promo_flag","price","dow","month","is_holiday","temp_z",
                "item_age","season"]
    STATIC   = ["category","subcategory","brand","vendor","format","region","elasticity",
                "base_price","pack_size","perishable","private_label","lead_time_days",
                "sqft","trade_area_income","gross_margin"]
    GRAPH    = PCOLS + SCOLS + ["sub_pagerank","sub_degree","subs_promo_pressure",
                "subs_neighbor_demand","subs_stockout_pressure","store_peer_demand"]

    df = df.dropna(subset=["y","lag_28"]).reset_index(drop=True)
    return df, dict(temporal=TEMPORAL, static=STATIC, graph=GRAPH), H


if __name__ == "__main__":
    df, feats, H = build_features()
    print("rows:", len(df), "| horizon:", H)
    print("feature blocks:", {k: len(v) for k, v in feats.items()})
    print("cold-start rows (item_age<90):", int((df.item_age < 90).sum()))
