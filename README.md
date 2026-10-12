# retail-demand-forecasting

Store x SKU x day replenishment forecasting with a product and store graph, and an
ablation that measures whether the graph is worth having.

## The problem

A warehouse-club retailer (called RetailCo here) reorders thousands of SKUs across dozens
of stores every day. Order too little and the shelf goes empty, which also hides the demand
that was lost, because the register only records what it could sell. Order too much and the
cash sits in the backroom. The hardest items are the new ones: a SKU launched last week has
no history of its own, so a per-item forecast has nothing to learn from exactly when the
buyer most needs a number.

## Why it is hard

The relationships that should help are well known to merchandisers but rarely reach the
model: a promoted substitute steals demand, a complement lifts it, a nearby store
cannibalises another, and a new item behaves like its neighbours in the assortment.
Published work on graph-augmented forecasting reports gains of around 16% over non-graph
baselines, largest on cold-start items. Whether that holds for a given retailer is an
empirical question, and stockouts censor the very demand the model is trying to learn.

## The approach

A simulator generates a two-year panel with the effects baked in by construction:
seasonality, price elasticity, promo lift, stockout censoring, substitution transfer and
store cannibalisation. Both the censored `units_sold` and the latent `true_demand` are
kept, so the pipeline can be checked against known ground truth. Product and store
relationships are modelled as a graph (an OWL/RDFS ontology loadable into Neo4j with
neosemantics), and the graph is turned into features: spectral (SVD) embeddings of
products and stores, substitution centrality, substitutes-on-promo pressure, and lagged
demand of substitutes and peer stores.

The key decision is that the graph is an ablated feature block, not the architecture. One
global LightGBM model (Tweedie loss for intermittent demand) is trained with and without
the graph features on the same split, so the graph has to earn its place in the numbers.
The target is forward 7-day demand, the quantity a replenishment order must cover, and a
p90 quantile head sizes safety stock.

## The result

DEMO scale: 6 stores, 200 SKUs, 730 days. The test window is the last 45 days, with a
7-day embargo before it to stop target leakage. Output of `python src/train.py`, captured
in `results/train_output.txt` (two runs produced identical numbers).

| Slice | Rows | Seasonal-naive WMAPE | LightGBM no-graph | LightGBM +graph | Graph lift |
|---|---|---|---|---|---|
| Overall | 55,200 | 0.2273 | 0.1838 | 0.1831 | +0.4% |
| Item age < 90 days | 8,736 | 0.2358 | 0.1965 | 0.1954 | +0.5% |
| Item age < 30 days | 3,030 | 0.2490 | 0.2040 | 0.2008 | +1.6% |
| Item age < 14 days | 1,662 | 0.2938 | 0.2439 | 0.2389 | +2.1% |

| Other measure | Value |
|---|---|
| Gradient boosting vs seasonal-naive, overall WMAPE | 0.2273 to 0.1838 (19% lower) |
| p90 head empirical coverage (target 0.90) | 0.885 |
| Graph features' share of total gain | 0.3% |

The disappointing number is the graph lift. The model beats the naive baseline clearly,
but the graph block adds 0.4% overall and 2.1% on the newest items, far short of the
published 16%, and the graph features carry 0.3% of the model's gain. Rolling 7- and
28-day means dominate. The lift does grow as items get newer, which is the direction the
literature predicts, but the newest slice covers only about a dozen SKUs launched inside
the test window, so the trend is suggestive rather than established. The p90 head slightly
under-covers.

## How it works

```
ontology/retail_demand.ttl   OWL/RDFS: Store, Product, Category, Vendor, Promotion, Region ...
        |  (optional) n10s import into Neo4j, see ontology/load_neo4j_n10s.cypher
        v
graph edges (substitute, complement, store-similar, store-near)
        |  truncated SVD of the normalised adjacency, centrality, neighbour demand
        v
graph features + temporal (lags, rolling stats, calendar, price, promo, item age) + static
        v
global LightGBM (Tweedie), with and without the graph block, plus a p90 quantile head
        v
forward 7-day demand per store x SKU  ->  order-up-to level
```

Neo4j is not needed to train. The embeddings are computed in-process with a deterministic
SVD; Neo4j GDS (node2vec or FastRP) over the same graph is a drop-in replacement.

## Running it

```bash
pip install -r requirements.txt
python src/generate_data.py   # writes data/*.csv, DEMO scale, about 0.9M rows
python src/train.py           # about 1 minute on a laptop; prints the tables above
```

Set `SCALE = "FULL"` in `src/config.py` for the 33-store, 4,000-SKU footprint. The
generated CSVs are not committed; they are reproduced from the fixed seed.

## The data

Entirely synthetic, from `src/generate_data.py`. `sales.csv` carries `units_sold`
(censored, what the register saw), `true_demand` (latent demand), `oos_flag`,
`promo_flag` and `price`. Dimension and edge files load straight into Neo4j with
`LOAD CSV`. Because the effects are simulated, a weak graph lift here says as much about
how strongly the simulator couples neighbouring items as about graph features in general.

## Project layout

```
ontology/   retail_demand.ttl (ontology), load_neo4j_n10s.cypher (Neo4j bootstrap + GDS recipe)
src/        config.py (scale switch), generate_data.py (simulator), features.py, train.py
results/    train_output.txt (captured run)
```

## Known limitations

- The biggest gap: the graph lift is small and the cold-start slices are too small to
  separate it from noise. There is no repeated-seed or rolling-origin backtest yet.
- Synthetic data only, at DEMO scale; FULL scale has not been trained.
- The target is built from `units_sold`, so stockout censoring is present in training and
  not corrected for.
- No automated tests yet.
- Embeddings are spectral SVD, not a trained GNN.

## Next

Strengthen substitution coupling in the simulator and check whether the lift follows, add
a rolling-origin backtest over several seeds, and correct for censoring using `oos_flag`.

## License

MIT
