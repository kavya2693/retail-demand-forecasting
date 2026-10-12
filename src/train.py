"""
Train & ablate the replenishment forecaster.

Compares three models on a time-based holdout (7-day embargo to stop target leakage):
  1. Seasonal-naive baseline      (7 * trailing-week mean)
  2. LightGBM  WITHOUT graph      (temporal + static)
  3. LightGBM  WITH graph         (+ SVD embeddings, centrality, neighbor demand)

Headline metric = WMAPE lift from the graph block, reported OVERALL and on the
NEW-ITEM COLD-START subset (item_age < 90d) — where Amazon reports the largest gain.
Also fits a p90 quantile head on the graph model for safety-stock coverage.

Run:  python src/train.py
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb
import features as F

def wmape(y, p):  return np.abs(y - p).sum() / np.abs(y).sum()
def rmse(y, p):   return float(np.sqrt(np.mean((y - p) ** 2)))
def bias(y, p):   return (p.sum() - y.sum()) / y.sum()

def fit(df_tr, df_va, cols, cat, params):
    dtr = lgb.Dataset(df_tr[cols], df_tr["y"], categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(df_va[cols], df_va["y"], reference=dtr, categorical_feature=cat, free_raw_data=False)
    return lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])

def main():
    df, feats, H = F.build_features()
    cat = [c for c in ["category","subcategory","brand","vendor","format","region","season"]]

    # ---- time-based split with embargo ----
    dmax = df["date"].max()
    test_start = dmax - pd.Timedelta(days=45)
    embargo = pd.Timedelta(days=H)
    tr = df[df["date"] < test_start - embargo].copy()
    va = df[(df["date"] >= test_start - embargo) & (df["date"] < test_start)].copy()  # tune slice
    te = df[df["date"] >= test_start].copy()
    print(f"split  train={len(tr):,}  valid={len(va):,}  test={len(te):,}  (test from {test_start.date()})")

    base_p = dict(objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.05,
                  num_leaves=63, min_child_samples=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, metric="rmse", verbosity=-1, seed=42)

    no_graph  = feats["temporal"] + feats["static"]
    with_graph = no_graph + feats["graph"]

    m_ng = fit(tr, va, no_graph, cat, base_p)
    m_wg = fit(tr, va, with_graph, cat, base_p)

    y = te["y"].values
    p_base = (7.0 * te["rmean_7"].clip(lower=0)).values
    p_ng = np.clip(m_ng.predict(te[no_graph], num_iteration=m_ng.best_iteration), 0, None)
    p_wg = np.clip(m_wg.predict(te[with_graph], num_iteration=m_wg.best_iteration), 0, None)

    age = te["item_age"].values
    def block(name, mask):
        yy = y[mask]
        print(f"\n== {name}  (n={int(mask.sum()):,}) ==")
        print(f"{'model':22s} {'WMAPE':>8s} {'RMSE':>8s} {'bias':>8s}")
        for lbl, p in [("seasonal-naive",p_base),("LightGBM no-graph",p_ng),("LightGBM +graph",p_wg)]:
            pp = p[mask]
            print(f"{lbl:22s} {wmape(yy,pp):8.4f} {rmse(yy,pp):8.2f} {bias(yy,pp):+8.3f}")
        lift = (wmape(yy,p_ng[mask]) - wmape(yy,p_wg[mask])) / wmape(yy,p_ng[mask])
        print(f"  -> graph WMAPE lift vs no-graph: {lift:+.1%}")
        return lift

    lift_all  = block("OVERALL", np.ones(len(y), bool))
    block("cold item_age<90", age < 90)
    block("cold item_age<30", age < 30)
    lift_cold = block("ULTRA-COLD item_age<14 (little/no own history)", age < 14)

    # ---- p90 safety-stock head on the graph model ----
    q_p = {**base_p, "objective":"quantile", "alpha":0.9}; q_p.pop("tweedie_variance_power",None)
    m_q = fit(tr, va, with_graph, cat, q_p)
    p90 = np.clip(m_q.predict(te[with_graph], num_iteration=m_q.best_iteration), 0, None)
    cover = (y <= p90).mean()
    print(f"\n== SAFETY STOCK (p90 quantile head) ==")
    print(f"empirical coverage P(y <= p90) = {cover:.3f}   (target 0.90)")
    print(f"mean order-up-to (p90) = {p90.mean():.2f} units / {H}d")

    # ---- feature importance (graph model) ----
    imp = pd.Series(m_wg.feature_importance("gain"), index=with_graph).sort_values(ascending=False)
    print("\ntop-12 features by gain:")
    print(imp.head(12).round(0).to_string())
    graph_share = imp[feats["graph"]].sum() / imp.sum()
    print(f"\ngraph-feature share of total gain: {graph_share:.1%}")

    print(f"\nSUMMARY: overall lift {lift_all:+.1%} | cold-start lift {lift_cold:+.1%} | p90 coverage {cover:.2f}")

if __name__ == "__main__":
    main()
