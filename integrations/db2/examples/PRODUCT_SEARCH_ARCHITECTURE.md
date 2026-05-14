
# Product Search Hybrid - Architecture & Data Flow

This document provides detailed architecture and sequence diagrams for the `product_search_hybrid.py` example, which demonstrates pure vector search with SQL filters and intelligent query parsing.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Component Diagram](#component-diagram)
3. [Data Flow Sequences](#data-flow-sequences)
4. [Intelligent Query Parser](#intelligent-query-parser)
5. [Search Scenarios](#search-scenarios)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Product Search Application                          │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                         Main Function                               │    │
│  │  • Initialize database connection                                   │    │
│  │  • Create source products table                                     │    │
│  │  • Load products from source table                                  │    │
│  │  • Setup vector store                                               │    │
│  │  • Index documents with embeddings                                  │    │
│  │  • Create search pipeline                                           │    │
│  │  • Execute search scenarios                                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    Intelligent Query Parser                         │    │
│  │  • Synonym mapping (running → marathon, jogging)                    │    │
│  │  • Brand extraction (Nike, Adidas, Merrell, etc.)                   │    │
│  │  • Color extraction (black, white, blue, etc.)                      │    │
│  │  • Price range parsing ($100-$200, under $150)                      │    │
│  │  • Stock status detection (in stock, available)                     │    │
│  │  • Rating extraction (4+ stars, 4.5+)                               │    │
│  │  • Filter generation (Haystack filter format)                       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Haystack Pipeline Layer                             │
│                                                                              │
│  ┌──────────────────────┐         ┌──────────────────────┐                 │
│  │  Indexing Pipeline   │         │   Search Pipeline    │                 │
│  │                      │         │                      │                 │
│  │  ┌────────────────┐ │         │  ┌────────────────┐ │                 │
│  │  │   Document     │ │         │  │  Text Embedder │ │                 │
│  │  │   Embedder     │ │         │  │  (Query)       │ │                 │
│  │  └────────┬───────┘ │         │  └────────┬───────┘ │                 │
│  │           │          │         │           │          │                 │
│  │           ▼          │         │           ▼          │                 │
│  │  ┌────────────────┐ │         │  ┌────────────────┐ │                 │
│  │  │   Document     │ │         │  │   Embedding    │ │                 │
│  │  │   Writer       │ │         │  │   Retriever    │ │                 │
│  │  └────────────────┘ │         │  └────────────────┘ │                 │
│  └──────────────────────┘         └──────────────────────┘                 │
│                                                                              │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DB2 Document Store Layer                                │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    DB2DocumentStore                                 │    │
│  │                                                                     │    │
│  │  Public Methods:                                                    │    │
│  │  • write_documents(documents, policy)                               │    │
│  │  • query_by_embedding(query_embedding, filters, top_k, offset)     │    │
│  │  • count_documents()                                                │    │
│  │  • filter_documents(filters)                                        │    │
│  │                                                                     │    │
│  │  Internal Methods:                                                  │    │
│  │  • _embedding_retrieval(query_embedding, top_k, filters, ...)      │    │
│  │  • _get_connection() [thread-local]                                │    │
│  │  • _validate_embedding_dimension(dimension)                         │    │
│  │  • _create_table_if_not_exists()                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                      Query Builder                                  │    │
│  │  • build_vector_search(metric, embedding, top_k, where, offset)    │    │
│  │  • build_insert_document(document_dict)                             │    │
│  │  • build_count_query(where_clause)                                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                      Filter Converter                               │    │
│  │  • convert_filters(filters) → (where_clause, params)                │    │
│  │  • Handle comparison operators ($lt, $gt, $eq, $in, etc.)           │    │
│  │  • Handle logical operators ($and, $or, $not)                       │    │
│  │  • Type casting (string vs numeric)                                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                      Converters                                     │    │
│  │  • document_to_db2_dict(document) → dict                            │    │
│  │  • db2_row_to_document(row) → Document                              │    │
│  │  • Score conversion: score = 1.0 - distance                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          IBM DB2 Database                                    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    source_products Table                            │    │
│  │  • product_id (VARCHAR PRIMARY KEY)                                 │    │
│  │  • name (VARCHAR)                                                   │    │
│  │  • description (VARCHAR)                                            │    │
│  │  • brand (VARCHAR)                                                  │    │
│  │  • category (VARCHAR)                                               │    │
│  │  • subcategory (VARCHAR)                                            │    │
│  │  • price (DECIMAL)                                                  │    │
│  │  • color (VARCHAR)                                                  │    │
│  │  • size (VARCHAR)                                                   │    │
│  │  • in_stock (BOOLEAN)                                               │    │
│  │  • rating (DECIMAL)                                                 │    │
│  │  • reviews (INTEGER)                                                │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    product_vectors Table                            │    │
│  │  • id (VARCHAR PRIMARY KEY)                                         │    │
│  │  • content (CLOB) - Rich product description                        │    │
│  │  • embedding (VECTOR(384, FLOAT32)) - Semantic vector               │    │
│  │  • meta (CLOB) - JSON: {brand, price, color, rating, ...}          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              product_vectors_metadata Table                         │    │
│  │  • table_name (VARCHAR PRIMARY KEY)                                 │    │
│  │  • embedding_model (VARCHAR)                                        │    │
│  │  • embedding_dimension (INTEGER)                                    │    │
│  │  • distance_metric (VARCHAR)                                        │    │
│  │  • created_at (TIMESTAMP)                                           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    Vector Operations                                │    │
│  │  • VECTOR_DISTANCE(v1, v2, COSINE)                                  │    │
│  │  • JSON_VALUE(meta, '$.field')                                      │    │
│  │  • CAST operations for numeric comparisons                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Product Search Components                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│   User Interface     │
│  (Terminal Output)   │
└──────────┬───────────┘
           │
           │ User Query: "black nike running shoes under $150"
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    parse_query_with_filters()                             │
│                                                                           │
│  Input: "black nike running shoes under $150"                            │
│                                                                           │
│  Processing:                                                              │
│  1. Apply synonym mapping                                                │
│     "running" → ["marathon", "jogging", "cardio"]                        │
│                                                                           │
│  2. Extract brand                                                         │
│     Pattern: \b(nike|adidas|merrell|...)\b                               │
│     Result: brand = "Nike"                                                │
│                                                                           │
│  3. Extract color                                                         │
│     Pattern: \b(black|white|blue|...)\b                                  │
│     Result: color = "black"                                               │
│                                                                           │
│  4. Extract price constraint                                              │
│     Pattern: under\s*\$?(\d+)                                            │
│     Result: price = {"$lt": 150}                                          │
│                                                                           │
│  5. Remove extracted terms from query                                     │
│     Remaining: "running shoes"                                            │
│                                                                           │
│  Output:                                                                  │
│  • semantic_query = "running shoes"                                       │
│  • filters = {                                                            │
│      "brand": "Nike",                                                     │
│      "color": "black",                                                    │
│      "price": {"$lt": 150}                                                │
│    }                                                                      │
└──────────────────────────────────────────────────────────────────────────┘
           │
           │ semantic_query + filters
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      Search Pipeline                                      │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Component 1: SentenceTransformersTextEmbedder                   │   │
│  │                                                                  │   │
│  │  Input: "running shoes"                                          │   │
│  │  Model: sentence-transformers/all-MiniLM-L6-v2                   │   │
│  │  Output: [0.123, -0.456, 0.789, ...] (384 dimensions)           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           │ query_embedding                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Component 2: DB2EmbeddingRetriever                              │   │
│  │                                                                  │   │
│  │  Input:                                                          │   │
│  │  • query_embedding: [0.123, -0.456, ...]                         │   │
│  │  • filters: {"brand": "Nike", "color": "black", ...}             │   │
│  │  • top_k: 3                                                      │   │
│  │                                                                  │   │
│  │  Processing:                                                     │   │
│  │  1. Call document_store.query_by_embedding()                     │   │
│  │  2. Apply filters before vector search                           │   │
│  │  3. Calculate cosine distance                                    │   │
│  │  4. Convert distance to similarity score                         │   │
│  │  5. Sort by score (descending)                                   │   │
│  │  6. Return top_k documents                                       │   │
│  │                                                                  │   │
│  │  Output: List[Document] with scores                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
           │
           │ documents with scores
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      search_products()                                    │
│                                                                           │
│  Processing:                                                              │
│  1. Apply score threshold (MIN_SCORE = 0.45)                             │
│  2. Filter out low-scoring documents                                     │
│  3. Format and display results                                           │
│                                                                           │
│  Output:                                                                  │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │ 1. Nike Pegasus Marathon                                        │     │
│  │    Brand: Nike                                                  │     │
│  │    Category: running                                            │     │
│  │    Price: $140.0                                                │     │
│  │    Color: black                                                 │     │
│  │    Rating: 4.7                                                  │     │
│  │    Similarity Score: 0.6752 (GREEN - high match)               │     │
│  │    Description: Nike Pegasus Marathon is a lightweight...      │     │
│  └────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Sequences

### Sequence 1: Initialization & Data Loading

```
┌──────┐         ┌──────────┐         ┌──────────┐         ┌──────────┐
│ Main │         │  ibm_db  │         │   DB2    │         │ Document │
│      │         │ (Python) │         │ Database │         │  Store   │
└──┬───┘         └────┬─────┘         └────┬─────┘         └────┬─────┘
   │                  │                    │                     │
   │ 1. Connect       │                    │                     │
   ├─────────────────>│                    │                     │
   │                  │                    │                     │
   │                  │ 2. TCP Connection  │                     │
   │                  ├───────────────────>│                     │
   │                  │                    │                     │
   │                  │ 3. Connection OK   │                     │
   │                  │<───────────────────┤                     │
   │                  │                    │                     │
   │ 4. Create source_products table       │                     │
   ├─────────────────>│                    │                     │
   │                  │                    │                     │
   │                  │ 5. CREATE TABLE    │                     │
   │                  ├───────────────────>│                     │
   │                  │                    │                     │
   │                  │ 6. Table Created   │                     │
   │                  │<───────────────────┤                     │
   │                  │                    │                     │
   │ 7. Insert 9 products (running, basketball, hiking, formal, winter)
   ├─────────────────>│                    │                     │
   │                  │                    │                     │
   │                  │ 8. INSERT INTO     │                     │
   │                  ├───────────────────>│                     │
   │                  │    (batch of 9)    │                     │
   │                  │                    │                     │
   │                  │ 9. Rows Inserted   │                     │
   │                  │<───────────────────┤                     │
   │                  │                    │                     │
   │ 10. Load products from source table   │                     │
   ├─────────────────>│                    │                     │
   │                  │                    │                     │
   │                  │ 11. SELECT * FROM  │                     │
   │                  ├───────────────────>│                     │
   │                  │                    │                     │
   │                  │ 12. Return 9 rows  │                     │
   │                  │<───────────────────┤                     │
   │                  │                    │                     │
   │ 13. Convert to Haystack Documents     │                     │
   │    (id, content=description, meta={brand, price, ...})      │
   │                  │                    │                     │
   │ 14. Initialize DB2DocumentStore       │                     │
   ├──────────────────────────────────────────────────────────>│
   │                  │                    │                     │
   │                  │                    │ 15. DROP TABLE     │
   │                  │                    │     product_vectors│
   │                  │                    │<────────────────────┤
   │                  │                    │                     │
   │                  │                    │ 16. CREATE TABLE   │
   │                  │                    │     product_vectors│
   │                  │                    │<────────────────────┤
   │                  │                    │                     │
   │                  │                    │ 17. CREATE TABLE   │
   │                  │                    │     ..._metadata   │
   │                  │                    │<────────────────────┤
   │                  │                    │                     │
   │ 18. Store initialized                 │                     │
   │<──────────────────────────────────────────────────────────┤
   │                  │                    │                     │
```

### Sequence 2: Document Indexing

```
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Main │    │ Indexing │    │ Document │    │ Document │    │   DB2    │
│      │    │ Pipeline │    │ Embedder │    │  Writer  │    │ Database │
└──┬───┘    └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
   │             │               │               │               │
   │ 1. Run indexing pipeline    │               │               │
   ├────────────>│               │               │               │
   │             │               │               │               │
   │             │ 2. Embed documents            │               │
   │             ├──────────────>│               │               │
   │             │               │               │               │
   │             │ For each document:            │               │
   │             │ • Load model (all-MiniLM-L6-v2)              │
   │             │ • Tokenize description        │               │
   │             │ • Generate 384-dim vector     │               │
   │             │               │               │               │
   │             │ 3. Documents with embeddings  │               │
   │             │<──────────────┤               │               │
   │             │               │               │               │
   │             │ 4. Write documents            │               │
   │             ├──────────────────────────────>│               │
   │             │               │               │               │
   │             │               │               │ 5. Validate   │
   │             │               │               │    model      │
   │             │               │               ├──────────────>│
   │             │               │               │               │
   │             │               │               │ 6. Check      │
   │             │               │               │    metadata   │
   │             │               │               │    table      │
   │             │               │               │               │
   │             │               │               │ 7. Store      │
   │             │               │               │    model info │
   │             │               │               │               │
   │             │               │               │ 8. INSERT     │
   │             │               │               │    documents  │
   │             │               │               │    (batch)    │
   │             │               │               │               │
   │             │               │               │ SQL:          │
   │             │               │               │ INSERT INTO   │
   │             │               │               │ product_vectors│
   │             │               │               │ (id, content, │
   │             │               │               │  embedding,   │
   │             │               │               │  meta)        │
   │             │               │               │ VALUES        │
   │             │               │               │ (?, ?, ?, ?)  │
   │             │               │               │               │
   │             │               │               │ 9. Rows       │
   │             │               │               │    inserted   │
   │             │               │               │<──────────────┤
   │             │               │               │               │
   │             │ 10. Write complete            │               │
   │             │<──────────────────────────────┤               │
   │             │               │               │               │
   │ 11. Indexing complete       │               │               │
   │<────────────┤               │               │               │
   │             │               │               │               │
```

### Sequence 3: Pure Vector Search (No Filters)

```
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ User │    │  Search  │    │   Text   │    │ Embedding│    │   DB2    │
│      │    │ Function │    │ Embedder │    │ Retriever│    │ Database │
└──┬───┘    └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
   │             │               │               │               │
   │ Query: "lightweight marathon running shoes" │               │
   ├────────────>│               │               │               │
   │             │               │               │               │
   │             │ 1. Embed query text           │               │
   │             ├──────────────>│               │               │
   │             │               │               │               │
   │             │ • Tokenize: ["lightweight", "marathon", ...]  │
   │             │ • Generate embedding vector   │               │
   │             │ • Output: [0.234, -0.567, ...] (384-dim)      │
   │             │               │               │               │
   │             │ 2. Query embedding            │               │
   │             │<──────────────┤               │               │
   │             │               │               │               │
   │             │ 3. Retrieve documents         │               │
   │             ├──────────────────────────────>│               │
   │             │               │               │               │
   │             │               │               │ 4. Build SQL  │
   │             │               │               ├──────────────>│
   │             │               │               │               │
   │             │               │               │ SQL:          │
   │             │               │               │ SELECT id,    │
   │             │               │               │   content,    │
   │             │               │               │   meta,       │
   │             │               │               │   VECTOR_     │
   │             │               │               │   DISTANCE(   │
   │             │               │               │     embedding,│
   │             │               │               │     CAST(     │
   │             │               │               │       '[...]' │
   │             │               │               │       AS      │
   │             │               │               │       VECTOR),│
   │             │               │               │     COSINE    │
   │             │               │               │   ) as distance│
   │             │               │               │ FROM          │
   │             │               │               │   product_    │
   │             │               │               │   vectors     │
   │             │               │               │ ORDER BY      │
   │             │               │               │   distance ASC│
   │             │               │               │ FETCH FIRST   │
   │             │               │               │   3 ROWS ONLY │
   │             │               │               │               │
   │             │               │               │ 5. Execute    │
   │             │               │               │    query      │
   │             │               │               │               │
   │             │               │               │ 6. Results:   │
   │             │               │               │ • Nike Pegasus│
   │             │               │               │   (dist=0.27) │
   │             │               │               │ • Adidas Boston│
   │             │               │               │   (dist=0.35) │
   │             │               │               │ • Nike Recovery│
   │             │               │               │   (dist=0.40) │
   │             │               │               │               │
   │             │               │               │ 7. Convert    │
   │             │               │               │    to docs    │
   │             │               │               │               │
   │             │               │               │ For each row: │
   │             │               │               │ • Parse JSON  │
   │             │               │               │   metadata    │
   │             │               │               │ • Calculate   │
   │             │               │               │   score =     │
   │             │               │               │   1 - distance│
   │             │               │               │ • Create      │
   │             │               │               │   Document    │
   │             │               │               │               │
   │             │               │               │ 8. Return docs│
   │             │               │               │<──────────────┤
   │             │               │               │               │
   │             │ 9. Documents with scores      │               │
   │             │<──────────────────────────────┤               │
   │             │               │               │               │
   │             │ Documents:                    │               │
   │             │ • Nike Pegasus (score=0.73)   │               │
   │             │ • Adidas Boston (score=0.65)  │               │
   │             │ • Nike Recovery (score=0.60)  │               │
   │             │               │               │               │
   │ 10. Display results         │               │               │
   │<────────────┤               │               │               │
   │             │               │               │               │
```

### Sequence 4: Hybrid Search (Vector + SQL Filters)

```
┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ User │    │  Parser  │    │  Search  │    │ Embedding│    │   DB2    │
│      │    │          │    │ Function │    │ Retriever│    │ Database │
└──┬───┘    └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
   │             │               │               │               │
   │ Query: "black nike running shoes under $150"               │
   ├────────────>│               │               │               │
   │             │               │               │               │
   │ 1. Parse query              │               │               │
   │             │               │               │               │
   │ Step 1: Apply synonyms      │               │               │
   │ • "running" → ["marathon", "jogging", "cardio"]            │
   │             │               │               │               │
   │ Step 2: Extract brand       │               │               │
   │ • Pattern: \b(nike|adidas|...)\b            │               │
   │ • Match: "nike" → "Nike"    │               │               │
   │ • Remove from query         │               │               │
   │             │               │               │               │
   │ Step 3: Extract color       │               │               │
   │ • Pattern: \b(black|white|...)\b            │               │
   │ • Match: "black"            │               │               │
   │ • Remove from query         │               │               │
   │             │               │               │               │
   │ Step 4: Extract price       │               │               │
   │ • Pattern: under\s*\$?(\d+) │               │               │
   │ • Match: "under $150" → 150 │               │               │
   │ • Create filter: {"$lt": 150}               │               │
   │ • Remove from query         │               │               │
   │             │               │               │               │
   │ 2. Parsed result:           │               │               │
   │ • semantic_query = "running shoes"          │               │
   │ • filters = {               │               │               │
   │     "brand": "Nike",        │               │               │
   │     "color": "black",       │               │               │
   │     "price": {"$lt": 150}   │               │               │
   │   }                         │               │               │
   │             │               │               │               │
   │ 3. Pass to search           │               │               │
   ├────────────────────────────>│               │               │
   │             │               │               │               │
   │             │ 4. Embed "running shoes"      │               │
   │             │               │ (same as pure vector search)  │
   │             │               │               │               │
   │             │ 5. Retrieve with filters      │               │
   │             │               ├──────────────>│               │
   │             │               │               │               │
   │             │               │               │ 6. Convert    │
   │             │               │               │    filters    │
   │             │               │               │               │
   │             │               │               │ Haystack:     │
   │             │               │               │ {             │
   │             │               │               │   "brand":    │
   │             │               │               │     "Nike",   │
   │             │               │               │   "color":    │
   │             │               │               │     "black",  │
   │             │               │               │   "price":    │
   │             │               │               │     {"$lt":150}│
   │             │               │               │ }             │
   │             │               │               │               │
   │             │               │               │ ↓             │
   │             │               │               │               │
   │             │               │               │ SQL WHERE:    │
   │             │               │               │ (JSON_VALUE(  │
   │             │               │               │   meta,       │
   │             │               │               │   '$.brand')  │
   │             │               │               │   = ?)        │
   │             │               │               │ AND           │
   │             │               │               │ (JSON_VALUE(  │
   │             │               │               │   meta,       │
   │             │               │               │   '$.color')  │
   │             │               │               │   = ?)        │
   │             │               │               │ AND           │
   │             │               │               │ (CAST(CAST(   │
   │             │               │               │   JSON_VALUE( │
   │             │               │               │     meta,     │
   │             │               │               │     '$.price')│
   │             │               │               │   AS VARCHAR) │
   │             │               │               │   AS DECFLOAT)│
   │             │               │               │   < CAST(? AS │
   │             │               │               │     DECFLOAT))│
   │             │               │               │               │
   │             │               │               │ Params:       │
   │             │               │               │ ['Nike',      │
   │             │               │               │  'black',     │
   │             │               │               │  150]         │
   │             │               │               │               │
   │             │               │               │ 7. Build SQL  │
   │             │               │               ├──────────────>│
   │             │               │               │               │
   │             │               │               │ SQL:          │
   │             │               │               │ SELECT id,    │
   │             │               │               │   content,    │
   │             │               │               │   meta,       │
   │             │               │               │   VECTOR_     │
   │             │               │               │   DISTANCE(...)│
   │             │               │               │     as distance│
   │             │               │               │ FROM          │
   │             │               │               │   product_    │
   │             │               │               │   vectors     │
   │             │               │               │ WHERE         │
   │             │               │               │   [filters]   │
   │             │               │               │ ORDER BY      │
   │             │               │               │   distance ASC│
   │             │               │               │ FETCH FIRST   │
   │             │               │               │   3 ROWS ONLY │
   │             │               │               │               │
   │             │               │               │ 8. Execute    │
   │             │               │               │    with params│
   │             │               │               │               │
   │             │               │               │ 9. Results:   │
   │             │               │               │ • Nike Pegasus│
   │             │               │               │   Marathon    │
   │             │               │               │   (black,     │
   │             │               │               │    $140,      │
   │             │               │               │    dist=0.32) │
   │             │               │               │               │
   │             │               │               │ (Only 1 match │
   │             │               │               │  meets all    │
   │             │               │               │  criteria)    │
   │             │               │               │               │
   │             │               │               │ 10. Return    │
   │             │               │               │<──────────────┤
   │             │               │               │               │
   │             │ 11. Documents with scores     │               │
   │             │<──────────────────────────────┤               │
   │             │               │               │               │
   │ 12. Display results         │               │               │
   │<────────────────────────────┤               │               │
   │             │               │               │               │
   │ Result:                     │               │               │
   │ 1. Nike Pegasus Marathon    │               │               │
   │    Brand: Nike              │               │               │
   │    Price: $140.0            │               │               │
   │    Color: black             │               │               │
   │    Similarity Score: 0.6752 │               │               │
   │             │               │               │               │
```

---

## Intelligent Query Parser

### Parser Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    parse_query_with_filters()                                │
│                                                                              │
│  Input: "black nike running shoes under $150 in stock"                      │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 1: Synonym Mapping                                           │    │
│  │                                                                     │    │
│  │  SYNONYM_MAP = {                                                    │    │
│  │    "running": ["marathon", "jogging", "cardio", "endurance"],      │    │
│  │    "basketball": ["court", "indoor", "hoops"],                     │    │
│  │    "hiking": ["trekking", "trail", "outdoor", "mountain"],         │    │
│  │    "formal": ["business", "office", "professional", "dress"],      │    │
│  │    "winter": ["snow", "cold", "freezing", "insulated"]             │    │
│  │  }                                                                  │    │
│  │                                                                     │    │
│  │  Process: Replace query terms with synonyms for better matching    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 2: Brand Extraction                                          │    │
│  │                                                                     │    │
│  │  SUPPORTED_BRANDS = [                                               │    │
│  │    "Nike", "Adidas", "Merrell", "Columbia", "Clarks", "Sorel"      │    │
│  │  ]                                                                  │    │
│  │                                                                     │    │
│  │  Pattern: \b(nike|adidas|merrell|columbia|clarks|sorel)\b          │    │
│  │  Flags: re.IGNORECASE                                               │    │
│  │                                                                     │    │
│  │  Match: "nike" → Extract "Nike"                                     │    │
│  │  Action: Remove "nike" from query                                   │    │
│  │  Filter: {"brand": "Nike"}                                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 3: Color Extraction                                          │    │
│  │                                                                     │    │
│  │  SUPPORTED_COLORS = [                                               │    │
│  │    "black", "white", "blue", "gray", "red", "brown"                │    │
│  │  ]                                                                  │    │
│  │                                                                     │    │
│  │  Pattern: \b(black|white|blue|gray|red|brown)\b                    │    │
│  │  Flags: re.IGNORECASE                                               │    │
│  │                                                                     │    │
│  │  Match: "black" → Extract "black"                                   │    │
│  │  Action: Remove "black" from query                                  │    │
│  │  Filter: {"color": "black"}                                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 4: Price Range Extraction                                    │    │
│  │                                                                     │    │
│  │  Pattern 1: \$?(\d+)\s*-\s*\$?(\d+)                                │    │
│  │  Example: "$100-$200" or "100-200"                                  │    │
│  │  Filter: {"price": {"$gte": 100, "$lte": 200}}                     │    │
│  │                                                                     │    │
│  │  Pattern 2: (?:under|below|less than)\s*\$?(\d+)                   │    │
│  │  Example: "under $150" or "below 150"                               │    │
│  │  Filter: {"price": {"$lt": 150}}                                    │    │
│  │                                                                     │    │
│  │  Pattern 3: (?:over|above|more than)\s*\$?(\d+)                    │    │
│  │  Example: "over $200" or "above 200"                                │    │
│  │  Filter: {"price": {"$gt": 200}}                                    │    │
│  │                                                                     │    │
│  │  Match: "under $150" → Extract 150                                  │    │
│  │  Action: Remove "under $150" from query                             │    │
│  │  Filter: {"price": {"$lt": 150}}                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 5: Stock Status Extraction                                   │    │
│  │                                                                     │    │
│  │  Pattern: (?:in stock|available)                                    │    │
│  │  Flags: re.IGNORECASE                                               │    │
│  │                                                                     │    │
│  │  Match: "in stock" → Extract True                                   │    │
│  │  Action: Remove "in stock" from query                               │    │
│  │  Filter: {"in_stock": True}                                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 6: Rating Extraction                                         │    │
│  │                                                                     │    │
│  │  Pattern: (\d+(?:\.\d+)?)\+?\s*(?:stars?|rating)                   │    │
│  │  Flags: re.IGNORECASE                                               │    │
│  │                                                                     │    │
│  │  Examples:                                                          │    │
│  │  • "4+ stars" → {"rating": {"$gte": 4.0}}                          │    │
│  │  • "4.5 rating" → {"rating": {"$gte": 4.5}}                        │    │
│  │                                                                     │    │
│  │  Match: "4+ stars" → Extract 4.0                                    │    │
│  │  Action: Remove "4+ stars" from query                               │    │
│  │  Filter: {"rating": {"$gte": 4.0}}                                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Stage 7: Cleanup & Finalization                                    │    │
│  │                                                                     │    │
│  │  1. Remove extra whitespace: re.sub(r"\s+", " ", query).strip()    │    │
│  │  2. If query is empty, default to "shoes"                           │    │
│  │  3. Combine all extracted filters                                   │    │
│  │  4. Return (semantic_query, filters)                                │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output:                                                                     │
│  • semantic_query = "running shoes"                                          │
│  • filters = {                                                               │
│      "brand": "Nike",                                                        │
│      "color": "black",                                                       │
│      "price": {"$lt": 150},                                                  │
│      "in_stock": True                                                        │
│    }                                                                         │

└─────────────────────────────────────────────────────────────────────────────┘
```

### Parser Flow Diagram

```
                    User Query
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   Stage 1: Synonym Mapping        │
        │   Replace terms with synonyms     │
        └───────────────┬───────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   Stage 2: Brand Extraction       │
        │   Pattern: \b(nike|adidas|...)\b  │
        │   Extract & Remove from query     │
        └───────────────┬───────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   Stage 3: Color Extraction       │
        │   Pattern: \b(black|white|...)\b  │
        │   Extract & Remove from query     │
        └───────────────┬───────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   Stage 4: Price Extraction       │
        │   Patterns:                       │
        │   • $100-$200 (range)             │
        │   • under $150 (upper limit)      │
        │   • over $200 (lower limit)       │
        │   Extract & Remove from query     │
        └───────────────┬───────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   Stage 5: Stock Extraction       │
        │   Pattern: in stock|available     │
        │   Extract & Remove from query     │
        └───────────────┬───────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   Stage 6: Rating Extraction      │
        │   Pattern: 4+ stars|4.5 rating    │
        │   Extract & Remove from query     │
        └───────────────┬───────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   Stage 7: Cleanup                │
        │   • Remove extra whitespace       │
        │   • Default to "shoes" if empty   │
        │   • Combine all filters           │
        └───────────────┬───────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────┐
        │   Output:                         │
        │   • semantic_query (string)       │
        │   • filters (dict)                │
        └───────────────────────────────────┘
```

---

## Search Scenarios

### Scenario 1: Pure Semantic Search

**Query**: `"lightweight marathon running shoes"`

**Processing**:
1. No filters extracted
2. Query embedded as-is
3. Vector search only

**SQL Generated**:
```sql
SELECT id, content, meta,
       VECTOR_DISTANCE(embedding, CAST('[...]' AS VECTOR(384, FLOAT32)), COSINE) as distance
FROM product_vectors
ORDER BY distance ASC
FETCH FIRST 3 ROWS ONLY
```

**Results**:
- Nike Pegasus Marathon (score: 0.7293)
- Adidas Boston Speed (score: 0.6507)
- Nike Recovery Comfort (score: 0.6015)

---

### Scenario 2: Hybrid Search with Single Filter

**Query**: `"running shoes"` + `filters={"price": {"$lt": 150}}`

**Processing**:
1. Query embedded
2. Filter converted to SQL WHERE clause
3. Vector search with pre-filtering

**SQL Generated**:
```sql
SELECT id, content, meta,
       VECTOR_DISTANCE(embedding, CAST('[...]' AS VECTOR(384, FLOAT32)), COSINE) as distance
FROM product_vectors
WHERE CAST(CAST(JSON_VALUE(meta, '$.price' RETURNING VARCHAR(1000)) AS VARCHAR(100)) AS DECFLOAT) < CAST(? AS DECFLOAT)
ORDER BY distance ASC
FETCH FIRST 3 ROWS ONLY
```

**Parameters**: `[150]`

**Results**:
- Nike Pegasus Marathon ($140, score: 0.6752)
- Clarks Executive Oxford ($145, score: 0.4539)

---

### Scenario 3: Intelligent Parsing with Multiple Filters

**Query**: `"black nike running shoes under $150"`

**Parser Output**:
- **Semantic Query**: `"running shoes"`
- **Filters**:
  ```python
  {
      "brand": "Nike",
      "color": "black",
      "price": {"$lt": 150}
  }
  ```

**SQL Generated**:
```sql
SELECT id, content, meta,
       VECTOR_DISTANCE(embedding, CAST('[...]' AS VECTOR(384, FLOAT32)), COSINE) as distance
FROM product_vectors
WHERE (JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?)
  AND (JSON_VALUE(meta, '$.color' RETURNING VARCHAR(1000)) = ?)
  AND (CAST(CAST(JSON_VALUE(meta, '$.price' RETURNING VARCHAR(1000)) AS VARCHAR(100)) AS DECFLOAT) < CAST(? AS DECFLOAT))
ORDER BY distance ASC
FETCH FIRST 3 ROWS ONLY
```

**Parameters**: `['Nike', 'black', 150]`

**Results**:
- Nike Pegasus Marathon (black, $140, score: 0.6752)

---

### Scenario 4: Complex Query with Stock Status

**Query**: `"waterproof hiking shoes in stock"`

**Parser Output**:
- **Semantic Query**: `"waterproof hiking shoes"`
- **Filters**:
  ```python
  {
      "in_stock": True
  }
  ```

**SQL Generated**:
```sql
SELECT id, content, meta,
       VECTOR_DISTANCE(embedding, CAST('[...]' AS VECTOR(384, FLOAT32)), COSINE) as distance
FROM product_vectors
WHERE JSON_VALUE(meta, '$.in_stock' RETURNING VARCHAR(1000)) = ?
ORDER BY distance ASC
FETCH FIRST 3 ROWS ONLY
```

**Parameters**: `['true']`

**Results**:
- Merrell Mountain Trek (score: 0.6445)
- Columbia Trail Explorer (score: 0.5832)
- Clarks Executive Oxford (score: 0.4847)

---

## Key Features Demonstrated

### 1. **Semantic Understanding**
- Query: "lightweight marathon running shoes"
- Matches: Nike Pegasus Marathon, Adidas Boston Speed
- **Why**: Embeddings capture semantic similarity between "marathon" and "long-distance", "lightweight" and "speed"

### 2. **SQL Filter Pre-filtering**
- Filters applied **BEFORE** vector distance calculation
- Reduces search space
- Improves performance
- Ensures business rules (price, stock, brand)

### 3. **Intelligent Query Parsing**
- Extracts structured filters from natural language
- Removes filter terms from semantic query
- Maintains clean separation: semantics vs. metadata

### 4. **Score Threshold Filtering**
- `MIN_SCORE = 0.45`
- Filters out low-relevance results
- Ensures quality over quantity

### 5. **Rich Product Descriptions**
- Metadata embedded in descriptions naturally
- Example: "Nike Pegasus Marathon is a lightweight running shoe priced at $140..."
- Enables semantic search on metadata values

---

## Performance Characteristics

### Indexing Performance
- **9 products**: ~2 seconds
- **Bottleneck**: Embedding generation (SentenceTransformers)
- **Optimization**: Batch processing (default: 100 documents)

### Search Performance
- **Pure Vector Search**: ~50-100ms
- **Hybrid Search (with filters)**: ~30-80ms (faster due to pre-filtering)
- **Query Parsing**: <1ms (regex-based)

### Scalability Considerations
1. **Vector Search**: O(n) complexity, scales linearly with table size
2. **Filter Pre-filtering**: Reduces n before vector calculation
3. **Pagination**: Supported via `offset` parameter
4. **Batch Indexing**: Configurable batch size for large datasets

---

## Comparison: Pure Vector vs. Hybrid

| Aspect | Pure Vector Search | Hybrid Search (Vector + Filters) |
|--------|-------------------|----------------------------------|
| **Query** | "running shoes" | "black nike running shoes under $150" |
| **Semantic Matching** | ✅ Yes | ✅ Yes |
| **Metadata Filtering** | ❌ No | ✅ Yes (SQL WHERE) |
| **Result Count** | 3 products | 1 product (highly targeted) |
| **Performance** | Slower (searches all) | Faster (pre-filtered) |
| **Use Case** | Exploratory search | Targeted search with constraints |
| **SQL Complexity** | Simple | Complex (WHERE + CAST) |

---

## Error Handling

### 1. **No Results Found**
```python
if not documents:
    print(colored("No matching products found", Colors.RED))
    return
```

### 2. **Score Threshold Filtering**
```python
MIN_SCORE = 0.45
documents = [doc for doc in documents if doc.score >= MIN_SCORE]
```

### 3. **Empty Query Handling**
```python
if not remaining_query:
    remaining_query = "shoes"  # Default fallback
```

### 4. **Invalid Filter Values**
- Parser validates brand against `SUPPORTED_BRANDS`
- Parser validates color against `SUPPORTED_COLORS`
- Price must be numeric
- Rating must be float

---

## Future Enhancements

### Planned Features
1. **Fuzzy Brand Matching**: Handle typos ("Nikee" → "Nike")
2. **Category Extraction**: Detect "running", "basketball", "hiking" as categories
3. **Size Extraction**: Parse "size 10", "US 10", "EU 44"
4. **Multi-language Support**: Extend parser for non-English queries
5. **Query Suggestions**: "Did you mean..." for misspellings
6. **Advanced Filters**: Date ranges, multiple brands, OR conditions

### Under Consideration
1. **Machine Learning Parser**: Replace regex with NER (Named Entity Recognition)
2. **Query Expansion**: Automatic synonym expansion
3. **Personalization**: User preference-based filtering
4. **A/B Testing**: Compare pure vector vs. hybrid effectiveness

---

## Troubleshooting

### Issue 1: Low Similarity Scores
**Symptom**: All scores below 0.5

**Causes**:
- Query too generic ("shoes")
- Products semantically dissimilar
- Wrong embedding model

**Solutions**:
- Use more specific queries
- Adjust `MIN_SCORE` threshold
- Verify embedding model consistency

### Issue 2: No Results with Filters
**Symptom**: Empty result set with filters

**Causes**:
- Filters too restrictive
- No products match all criteria
- Filter values incorrect (case sensitivity)

**Solutions**:
- Relax filters (remove one constraint)
- Check filter values match metadata
- Use OR logic instead of AND

### Issue 3: Parser Not Extracting Filters
**Symptom**: Filters remain in semantic query

**Causes**:
- Pattern mismatch (typo in brand/color)
- Unsupported filter type
- Regex pattern needs update

**Solutions**:
- Add brand/color to supported lists
- Update regex patterns
- Check for typos in query

---

## Summary

The `product_search_hybrid.py` example demonstrates a production-ready product search system with:

1. **Pure Vector Search**: Semantic similarity matching
2. **Hybrid Search**: Vector + SQL filters for targeted results
3. **Intelligent Parsing**: Natural language → structured filters
4. **Score Thresholding**: Quality control
5. **Rich Descriptions**: Metadata embedded naturally

**Key Takeaway**: Combining vector search with SQL filters provides the best of both worlds - semantic understanding with structured constraints.

---

**Last Updated**: 2026-05-10  
**Example File**: `product_search_hybrid.py`  
**Lines of Code**: 1223  
**Search Scenarios**: 18 (6 pure vector + 4 hybrid + 8 intelligent parsing)
