"""
RetailCo graph-augmented demand simulator.

Emits, into data/:
  dimensions          stores.csv, products.csv, calendar.csv
  graph edges         edges_substitute.csv, edges_complement.csv,
                      edges_store_similar.csv, edges_store_near.csv
  fact table          sales.csv   (store x SKU x day)

The fact table carries the *ground-truth* signals a graph model is meant to
recover, so the forecasting pipeline can be validated against known effects:
  - true_demand   : latent demand before shelf availability
  - units_sold    : censored observation (== true_demand unless OOS that day)
  - oos_flag      : shelf was empty -> lost sales (the replenishment problem)
  - promo_flag / price : own price & promo lever
  - Substitution transfer and store cannibalization are baked into true_demand.

Run:  python src/generate_data.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

import config as C

rng = np.random.default_rng(C.SEED)
C.DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. DIMENSIONS
# --------------------------------------------------------------------------- #
def build_stores() -> pd.DataFrame:
    regions = ["West", "Southwest", "Midwest", "Northeast", "Southeast"]
    fmts = ["Warehouse", "Business Center"]
    # cluster stores around a few metro centroids so NEAR/geo-cannibalization is real
    centroids = np.array([[34.0,-118.2],[37.8,-122.4],[41.9,-87.6],[40.7,-74.0],[33.7,-84.4]])
    ci = rng.integers(0, len(centroids), C.N_STORES)
    lat = centroids[ci,0] + rng.normal(0, 0.6, C.N_STORES)
    lon = centroids[ci,1] + rng.normal(0, 0.6, C.N_STORES)
    df = pd.DataFrame({
        "store_id": [f"S{ i:03d}" for i in range(C.N_STORES)],
        "region": [regions[k % len(regions)] for k in ci],
        "format": rng.choice(fmts, C.N_STORES, p=[0.85, 0.15]),
        "sqft": rng.integers(90_000, 165_000, C.N_STORES),
        "lat": lat.round(4), "lon": lon.round(4),
        "trade_area_income": rng.integers(55_000, 145_000, C.N_STORES),
        "open_date": pd.to_datetime("2010-01-01") + pd.to_timedelta(rng.integers(0, 4500, C.N_STORES), "D"),
    })
    df["open_date"] = df["open_date"].dt.date
    return df


def _launch_dates() -> np.ndarray:
    start = pd.to_datetime(C.START_DATE)
    is_new = rng.random(C.N_PRODUCTS) < C.NEW_ITEM_SHARE
    is_late = is_new & (rng.random(C.N_PRODUCTS) < C.NEW_ITEM_LATE_FRAC)
    out = start - pd.to_timedelta(rng.integers(30, 1500, C.N_PRODUCTS), "D")          # mature default
    early = is_new & ~is_late
    out = np.where(early, start + pd.to_timedelta(rng.integers(60, C.N_DAYS-160, C.N_PRODUCTS), "D"), out)
    out = np.where(is_late, start + pd.to_timedelta(rng.integers(C.N_DAYS-130, C.N_DAYS-15, C.N_PRODUCTS), "D"), out)
    return pd.to_datetime(out)


def build_products() -> pd.DataFrame:
    subcats = [(cat, sub) for cat, subs in C.CATEGORIES.items() for sub in subs]
    pick = rng.integers(0, len(subcats), C.N_PRODUCTS)
    cat = [subcats[i][0] for i in pick]
    sub = [subcats[i][1] for i in pick]
    brand = [f"{c[:3].upper()}-B{rng.integers(0, C.BRANDS_PER_CAT)}" for c in cat]
    base_price = np.round(rng.lognormal(2.4, 0.7, C.N_PRODUCTS), 2).clip(1.5, 900)
    intermittent = rng.random(C.N_PRODUCTS) < C.INTERMITTENT_SHARE
    df = pd.DataFrame({
        "sku_id": [f"P{i:05d}" for i in range(C.N_PRODUCTS)],
        "name": [f"{s} item {i}" for i, s in enumerate(sub)],
        "category": cat, "subcategory": sub, "brand": brand,
        "vendor": [f"V{rng.integers(0,40):02d}" for _ in range(C.N_PRODUCTS)],
        "lead_time_days": rng.choice([2,3,5,7,10,14], C.N_PRODUCTS),
        "base_price": base_price,
        "unit_cost": np.round(base_price * rng.uniform(0.6, 0.85, C.N_PRODUCTS), 2),
        "pack_size": rng.choice([1,2,4,6,12,24], C.N_PRODUCTS),
        "perishable": [c == "Fresh" for c in cat],
        "private_label": rng.random(C.N_PRODUCTS) < 0.25,
        "elasticity": np.round(-rng.uniform(0.6, 2.6, C.N_PRODUCTS), 2),  # negative
        # popularity governs baseline volume; intermittent items get low popularity
        "_popularity": np.where(intermittent,
                                rng.uniform(0.05, 0.6, C.N_PRODUCTS),
                                rng.uniform(1.0, 6.0, C.N_PRODUCTS)),
        "_intermittent": intermittent,
        # mature (before panel) vs NEW items launched mid-panel -> real cold-start.
        # New items are concentrated into the last ~130 days so they populate the test window.
        "launch_date": _launch_dates(),
        # weather sensitivity: Seasonal + Beverages react to temperature
        "_temp_sensitivity": np.where(np.isin(sub, ["Summer","Beverages"]),
                                      rng.uniform(0.15, 0.35, C.N_PRODUCTS), 0.0),
    })
    df["launch_date"] = df["launch_date"].dt.date
    return df


def build_calendar() -> pd.DataFrame:
    dates = pd.date_range(C.START_DATE, periods=C.N_DAYS, freq="D")
    doy = dates.dayofyear.values
    # simple US-ish holiday set (no external dep): New Year, Jul 4, Thanksgiving-ish, Christmas
    holiday = ((dates.month==1)&(dates.day==1)) | ((dates.month==7)&(dates.day==4)) | \
              ((dates.month==11)&(dates.day.isin([25,26,27,28,29]))) | \
              ((dates.month==12)&(dates.day.isin([24,25,26])))
    season = pd.cut(dates.month, [0,2,5,8,11,12],
                    labels=["Winter","Spring","Summer","Fall","Winter"], ordered=False)
    df = pd.DataFrame({
        "date": dates, "dow": dates.dayofweek, "doy": doy,
        "month": dates.month, "is_holiday": holiday.astype(int),
        "season": season.astype(str),
        # smooth seasonal temperature curve, peak in July
        "temp_z": np.round(np.sin((doy-15)/365*2*np.pi - np.pi/2) , 3),
    })
    return df


# --------------------------------------------------------------------------- #
# 2. GRAPH EDGES
# --------------------------------------------------------------------------- #
def build_product_edges(prod: pd.DataFrame):
    subs, comps = [], []
    by_sub = prod.groupby("subcategory").indices
    idx_all = np.arange(len(prod))
    for i, row in prod.iterrows():
        # substitutes: same subcategory, other SKUs
        peers = [j for j in by_sub[row.subcategory] if j != i]
        for j in rng.choice(peers, min(C.SUBS_PER_PRODUCT, len(peers)), replace=False) if peers else []:
            subs.append((row.sku_id, prod.iloc[j].sku_id, round(float(rng.uniform(0.3,0.9)),3)))
        # complements: any category, biased to different category
        for j in rng.choice(idx_all, C.COMPLEMENTS_PER_PRODUCT, replace=False):
            if prod.iloc[j].sku_id != row.sku_id:
                comps.append((row.sku_id, prod.iloc[j].sku_id, round(float(rng.uniform(0.2,0.7)),3)))
    subs = pd.DataFrame(subs, columns=["sku_a","sku_b","strength"]).drop_duplicates(["sku_a","sku_b"])
    comps = pd.DataFrame(comps, columns=["sku_a","sku_b","strength"]).drop_duplicates(["sku_a","sku_b"])
    return subs, comps


def haversine(lat1, lon1, lat2, lon2):
    R=6371.0; p=np.pi/180
    a=(np.sin((lat2-lat1)*p/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2)
    return 2*R*np.arcsin(np.sqrt(a))


def build_store_edges(stores: pd.DataFrame):
    n = len(stores); sim_rows=[]; near_rows=[]
    attr = stores[["sqft","trade_area_income"]].to_numpy(float)
    attr = (attr - attr.mean(0)) / attr.std(0)
    for i in range(n):
        d = np.linalg.norm(attr - attr[i], axis=1); d[i]=np.inf
        for j in np.argsort(d)[:C.STORE_SIMILAR_K]:
            sim_rows.append((stores.store_id[i], stores.store_id[j], round(float(1/(1+d[j])),3)))
        for j in range(n):
            if j==i: continue
            km = haversine(stores.lat[i], stores.lon[i], stores.lat[j], stores.lon[j])
            if km <= C.STORE_NEAR_KM:
                near_rows.append((stores.store_id[i], stores.store_id[j], round(float(km),2)))
    return (pd.DataFrame(sim_rows, columns=["store_a","store_b","similarity"]),
            pd.DataFrame(near_rows, columns=["store_a","store_b","distance_km"]))


# --------------------------------------------------------------------------- #
# 3. SALES FACT  (store x SKU x day)  — vectorised per store-SKU series
# --------------------------------------------------------------------------- #
def _sub_matrix(prod, subs):
    """S[a,b] = strength that substitute b diverts demand to/from focal a."""
    idx = {s: i for i, s in enumerate(prod["sku_id"])}
    n = len(prod); S = np.zeros((n, n), np.float32)
    for _, e in subs.iterrows():
        a, b = idx.get(e.sku_a), idx.get(e.sku_b)
        if a is not None and b is not None:
            S[a, b] = e.strength
    return S


def build_sales(stores, prod, cal, subs):
    days = cal["date"].values
    D = len(cal)
    dow = cal["dow"].values
    temp = cal["temp_z"].values
    holiday = cal["is_holiday"].values
    doy = cal["doy"].values

    # day-of-week multiplier: weekend heavy (warehouse club pattern)
    dow_mult = np.array([0.85,0.8,0.85,0.95,1.15,1.5,1.35])[dow]
    # annual seasonality shared component
    year_wave = 1 + 0.15*np.sin((doy-60)/365*2*np.pi)

    # store size multiplier + substitution adjacency (focal a <- substitute b)
    store_mult = (stores["sqft"].to_numpy() / stores["sqft"].mean())
    S = _sub_matrix(prod, subs)                      # (P,P)
    launch = pd.to_datetime(prod["launch_date"]).values.astype("datetime64[D]")
    days64 = days.astype("datetime64[D]")
    launched = (days64[None, :] >= launch[:, None])  # (P,D) bool, shared across stores

    frames = []
    promo_lift_track = []  # (was_promo, demand)
    for si, store in stores.iterrows():
        base = (prod["_popularity"].to_numpy() * store_mult[si]
                * (0.7 + 0.6*rng.random(len(prod))))            # store-SKU idiosyncrasy

        # ---- price & promo (store-SKU-day) ----
        promo = (rng.random((len(prod), D)) < C.PROMO_RATE)
        depth = np.where(promo, rng.uniform(0.15, 0.45, (len(prod), D)), 0.0)
        price = prod["base_price"].to_numpy()[:,None] * (1 - depth)

        # ---- own-item demand (multiplicative) ----
        elas = prod["elasticity"].to_numpy()[:,None]
        price_effect = (price / prod["base_price"].to_numpy()[:,None]) ** elas
        promo_extra = np.where(promo, rng.uniform(C.PROMO_MIN_LIFT, C.PROMO_MAX_LIFT, (len(prod),D)), 1.0)
        temp_effect = 1 + prod["_temp_sensitivity"].to_numpy()[:,None]*temp[None,:]
        hol_effect  = 1 + 0.4*holiday[None,:]
        lam = (base[:,None] * dow_mult[None,:] * year_wave[None,:]
               * price_effect * promo_extra * temp_effect * hol_effect) * launched

        # ---- PASS 1: own demand + OOS realization (yields lost demand per item/day) ----
        td_base = rng.poisson(np.clip(lam, 0.0, None))
        oos = rng.random((len(prod), D)) < C.OOS_BASE_RATE
        cap = np.where(oos, rng.integers(0, 2, (len(prod), D)), 10**6)
        lost_base = np.maximum(td_base - np.minimum(td_base, cap), 0)     # censored units (P,D)

        # ---- PASS 2: substitution graph transfer (the graph signal) ----
        # stockout substitution: substitute b OOS -> its lost demand spills ONTO focal a (UP)
        t_up   = S @ (C.SUBS_STOCKOUT_TRANSFER * lost_base)               # (P,D)
        # promo cannibalization: substitute b on promo -> pulls demand OFF focal a (DOWN)
        promo_boost = np.clip(lam * (1 - 1/np.maximum(promo_extra, 1e-9)), 0, None)
        t_down = S @ (C.SUBS_PROMO_TRANSFER * promo_boost)                # (P,D)
        extra  = rng.poisson(np.clip(t_up, 0.0, None)) * launched
        true_demand = np.clip(td_base + extra - t_down, 0, None).round().astype(int)

        # ---- final censoring under the same shelf capacity ----
        units_sold = np.minimum(true_demand, cap)
        oos_flag = ((units_sold < true_demand) | oos).astype(int)

        # ---- long-format frame for this store ----
        P, T = np.meshgrid(np.arange(len(prod)), np.arange(D), indexing="ij")
        frames.append(pd.DataFrame({
            "store_id": store.store_id,
            "sku_id": prod["sku_id"].to_numpy()[P.ravel()],
            "date": days[T.ravel()],
            "units_sold": units_sold.ravel(),
            "true_demand": true_demand.ravel(),
            "oos_flag": oos_flag.ravel(),
            "promo_flag": promo.ravel().astype(int),
            "price": np.round(price, 2).ravel(),
        }))
        promo_lift_track.append((promo.ravel(), true_demand.ravel(), base[:,None].repeat(D,1).ravel()))
    sales = pd.concat(frames, ignore_index=True)
    return sales, promo_lift_track


# --------------------------------------------------------------------------- #
def main():
    print(f"[gen] scale={C.SCALE}  {C.N_STORES} stores x {C.N_PRODUCTS} SKU x {C.N_DAYS} days")
    stores = build_stores();  stores.to_csv(C.DATA_DIR/"stores.csv", index=False)
    prod   = build_products()
    cal    = build_calendar(); cal.to_csv(C.DATA_DIR/"calendar.csv", index=False)

    subs, comps = build_product_edges(prod)
    subs.to_csv(C.DATA_DIR/"edges_substitute.csv", index=False)
    comps.to_csv(C.DATA_DIR/"edges_complement.csv", index=False)
    sim, near = build_store_edges(stores)
    sim.to_csv(C.DATA_DIR/"edges_store_similar.csv", index=False)
    near.to_csv(C.DATA_DIR/"edges_store_near.csv", index=False)

    # new items inherit base popularity from their substitute neighbors -> neighbor
    # demand is genuinely predictive for cold-start (beyond category averages)
    start = pd.to_datetime(C.START_DATE)
    is_new = pd.to_datetime(prod["launch_date"]) >= start
    idx = {s: i for i, s in enumerate(prod["sku_id"])}
    pop = prod["_popularity"].to_numpy().copy()
    nbrs = subs.groupby("sku_a")["sku_b"].apply(list).to_dict()
    for i, sku in enumerate(prod["sku_id"]):
        if is_new.iloc[i] and sku in nbrs:
            js = [idx[b] for b in nbrs[sku] if b in idx]
            if js:
                pop[i] = float(np.mean(prod["_popularity"].to_numpy()[js])) * rng.uniform(0.7, 1.3)
    prod["_popularity"] = pop

    # write products without helper cols
    prod.drop(columns=[c for c in prod.columns if c.startswith("_")]).to_csv(
        C.DATA_DIR/"products.csv", index=False)

    sales, track = build_sales(stores, prod, cal, subs)
    sales.to_csv(C.DATA_DIR/"sales.csv", index=False)

    # ---- effect sanity checks ----
    p = sales["promo_flag"].astype(bool)
    lift = sales.loc[p,"true_demand"].mean() / sales.loc[~p,"true_demand"].mean()
    censored = (sales["units_sold"] < sales["true_demand"]).mean()
    print(f"[gen] wrote sales.csv rows={len(sales):,}")
    print(f"[gen] edges: subs={len(subs)} comp={len(comps)} store_sim={len(sim)} store_near={len(near)}")
    print(f"[check] promo avg-lift x{lift:.2f}  (expect >1.5)")
    print(f"[check] censored share {censored:.3%}  (OOS lost-sales days)")
    print(f"[check] zero-sales share {(sales['units_sold']==0).mean():.1%}  (intermittency)")
    n_new = int((pd.to_datetime(prod['launch_date']) >= pd.to_datetime(C.START_DATE)).sum())
    print(f"[check] new-item SKUs (cold-start): {n_new} of {len(prod)}")
    print(f"[check] files in {C.DATA_DIR}")

if __name__ == "__main__":
    main()
