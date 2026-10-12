"""Central config for the RetailCo graph-augmented demand simulator.

DEMO scale keeps a full 2-year panel tractable on a laptop while preserving every
effect (seasonality, elasticity, promo lift, OOS censoring, substitution transfer,
store cannibalization). Flip to FULL for the full 33-store, 4,000-SKU footprint.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SEED = 42

# ---- Scale ------------------------------------------------------------------
# DEMO: runnable in seconds, ~0.9M rows.  FULL: 33 stores x 4000 SKU = target footprint.
SCALE = "DEMO"
_SCALES = {
    "DEMO": dict(n_stores=6,  n_products=200,  n_days=730),
    "FULL": dict(n_stores=33, n_products=4000, n_days=730),
}
N_STORES   = _SCALES[SCALE]["n_stores"]
N_PRODUCTS = _SCALES[SCALE]["n_products"]
N_DAYS     = _SCALES[SCALE]["n_days"]
START_DATE = "2024-01-01"

# ---- Assortment taxonomy ----------------------------------------------------
CATEGORIES = {
    "Grocery":    ["Snacks", "Beverages", "Pantry", "Coffee"],
    "Fresh":      ["Produce", "Meat", "Bakery", "Dairy"],
    "Household":  ["Cleaning", "Paper", "Storage"],
    "Electronics":["TV", "Audio", "Accessories"],
    "Apparel":    ["Outerwear", "Basics"],
    "Seasonal":   ["Summer", "Holiday"],
}
BRANDS_PER_CAT = 6

# ---- Graph density ----------------------------------------------------------
SUBS_PER_PRODUCT       = 3     # substitution edges (within subcategory)
COMPLEMENTS_PER_PRODUCT = 2    # complement edges (cross-category affinity)
STORE_SIMILAR_K        = 3     # k nearest-by-attribute store peers
STORE_NEAR_KM          = 60.0  # NEAR edge if within this many km

# ---- Demand dynamics --------------------------------------------------------
PROMO_RATE          = 0.06   # share of store-SKU-days on promo
PROMO_MIN_LIFT      = 1.6    # multiplicative demand lift floor during promo
PROMO_MAX_LIFT      = 3.2
SUBS_PROMO_TRANSFER    = 0.6  # promo cannibalization: promo on b pulls demand OFF its substitutes
SUBS_STOCKOUT_TRANSFER = 1.1  # stockout substitution: b OOS pushes its lost demand ONTO substitutes
OOS_BASE_RATE       = 0.09   # baseline probability a store-SKU is out of stock on a day (raised: drives substitution)
INTERMITTENT_SHARE  = 0.45   # share of SKUs that are slow/lumpy movers
NEW_ITEM_SHARE      = 0.25   # share of SKUs that launch mid-panel (cold-start)
NEW_ITEM_LATE_FRAC  = 0.6    # of new items, share launched into the last ~130 days (test window)

if __name__ == "__main__":
    rows = N_STORES * N_PRODUCTS * N_DAYS
    print(f"SCALE={SCALE}  stores={N_STORES} products={N_PRODUCTS} days={N_DAYS}  ->  {rows:,} panel rows")
