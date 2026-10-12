// ============================================================
//  RetailCo Retail Demand — Neo4j + neosemantics (n10s) bootstrap
//  Run in cypher-shell / Neo4j Browser against a DB with the
//  neosemantics and Graph Data Science (GDS) plugins installed.
// ============================================================

// ---- 1. n10s config + unique constraint on resources ----
CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS
  FOR (r:Resource) REQUIRE r.uri IS UNIQUE;

CALL n10s.graphconfig.init({
  handleVocabUris: 'IGNORE',        // strip namespace, keep short labels (Store, Product, ...)
  handleRDFTypes:  'LABELS',        // owl:Class instances -> node labels
  keepLangTag:     false,
  applyNeo4jNaming: true
});

// ---- 2. Import the OWL/RDFS ontology (schema layer) ----
// Serve retail_demand.ttl over http(s) or file:// reachable by the DB.
CALL n10s.onto.import.fetch(
  'file:///import/retail_demand.ttl', 'Turtle'
);

// ---- 3. Business-key constraints (data layer, loaded by ETL/generator) ----
CREATE CONSTRAINT store_id  IF NOT EXISTS FOR (s:Store)    REQUIRE s.storeId IS UNIQUE;
CREATE CONSTRAINT sku_id    IF NOT EXISTS FOR (p:Product)  REQUIRE p.skuId   IS UNIQUE;
CREATE CONSTRAINT cat_id    IF NOT EXISTS FOR (c:Category) REQUIRE c.name    IS UNIQUE;
CREATE CONSTRAINT brand_id  IF NOT EXISTS FOR (b:Brand)    REQUIRE b.name    IS UNIQUE;
CREATE CONSTRAINT vendor_id IF NOT EXISTS FOR (v:Vendor)   REQUIRE v.name    IS UNIQUE;

// ---- 4. Load instance data from the generator's CSVs (LOAD CSV) ----
// Products
LOAD CSV WITH HEADERS FROM 'file:///products.csv' AS row
MERGE (p:Product {skuId: row.sku_id})
  SET p.productName = row.name, p.basePrice = toFloat(row.base_price),
      p.unitCost = toFloat(row.unit_cost), p.packSize = toInteger(row.pack_size),
      p.perishable = row.perishable = 'True', p.privateLabel = row.private_label = 'True',
      p.elasticity = toFloat(row.elasticity), p.launchDate = date(row.launch_date)
MERGE (c:Category {name: row.subcategory})
MERGE (parent:Category {name: row.category})
MERGE (c)-[:SUBCATEGORY_OF]->(parent)
MERGE (b:Brand {name: row.brand})
MERGE (v:Vendor {name: row.vendor}) SET v.leadTimeDays = toInteger(row.lead_time_days)
MERGE (p)-[:IN_CATEGORY]->(c)
MERGE (p)-[:HAS_BRAND]->(b)
MERGE (p)-[:SUPPLIED_BY]->(v);

// Stores
LOAD CSV WITH HEADERS FROM 'file:///stores.csv' AS row
MERGE (s:Store {storeId: row.store_id})
  SET s.format = row.format, s.sqft = toInteger(row.sqft),
      s.latitude = toFloat(row.lat), s.longitude = toFloat(row.lon),
      s.tradeAreaIncome = toInteger(row.trade_area_income), s.openDate = date(row.open_date)
MERGE (r:Region {name: row.region})
MERGE (s)-[:LOCATED_IN]->(r);

// Cross-product edges
LOAD CSV WITH HEADERS FROM 'file:///edges_substitute.csv' AS row
MATCH (a:Product {skuId: row.sku_a}), (b:Product {skuId: row.sku_b})
MERGE (a)-[e:SUBSTITUTE_FOR]->(b) SET e.strength = toFloat(row.strength);

LOAD CSV WITH HEADERS FROM 'file:///edges_complement.csv' AS row
MATCH (a:Product {skuId: row.sku_a}), (b:Product {skuId: row.sku_b})
MERGE (a)-[e:COMPLEMENT_OF]->(b) SET e.strength = toFloat(row.strength);

// Cross-store edges
LOAD CSV WITH HEADERS FROM 'file:///edges_store_similar.csv' AS row
MATCH (a:Store {storeId: row.store_a}), (b:Store {storeId: row.store_b})
MERGE (a)-[e:SIMILAR_TO]->(b) SET e.similarity = toFloat(row.similarity);

LOAD CSV WITH HEADERS FROM 'file:///edges_store_near.csv' AS row
MATCH (a:Store {storeId: row.store_a}), (b:Store {storeId: row.store_b})
MERGE (a)-[e:NEAR]->(b) SET e.distance_km = toFloat(row.distance_km);

// ============================================================
//  5. GRAPH-DERIVED FEATURES via GDS (no PyTorch Geometric needed)
//     These embeddings/scores are exported and joined to the
//     store x SKU x day panel as LightGBM features.
// ============================================================

// 5a. Project the product substitution/complement subgraph
CALL gds.graph.project(
  'productGraph',
  'Product',
  {
    SUBSTITUTE_FOR: {properties: 'strength', orientation: 'UNDIRECTED'},
    COMPLEMENT_OF:  {properties: 'strength', orientation: 'UNDIRECTED'}
  }
);

// 5b. node2vec embeddings per product (cross-product structure)
CALL gds.node2vec.write('productGraph', {
  embeddingDimension: 32, walkLength: 20, iterations: 3,
  relationshipWeightProperty: 'strength', writeProperty: 'emb_prod'
});

// 5c. FastRP as a fast alternative embedding
CALL gds.fastRP.write('productGraph', {
  embeddingDimension: 32, relationshipWeightProperty: 'strength',
  writeProperty: 'emb_prod_fastrp'
});

// 5d. Centrality = cold-start / hub proxy (isolated new SKUs score low)
CALL gds.pageRank.write('productGraph', {writeProperty: 'pagerank_sub'});
CALL gds.degree.write('productGraph', {writeProperty: 'sub_degree'});

// 5e. Store similarity graph -> store embeddings (spatial signal)
CALL gds.graph.project('storeGraph', 'Store',
  { SIMILAR_TO: {properties: 'similarity', orientation: 'UNDIRECTED'},
    NEAR:       {properties: 'distance_km', orientation: 'UNDIRECTED'} });
CALL gds.fastRP.write('storeGraph', {
  embeddingDimension: 16, relationshipWeightProperty: 'similarity',
  writeProperty: 'emb_store'
});

// 5f. Export node embeddings for the ML pipeline
//     (run as separate queries; stream to Python via the neo4j driver)
// MATCH (p:Product) RETURN p.skuId AS sku_id, p.emb_prod AS emb, p.pagerank_sub AS pr, p.sub_degree AS deg;
// MATCH (s:Store)   RETURN s.storeId AS store_id, s.emb_store AS emb;
