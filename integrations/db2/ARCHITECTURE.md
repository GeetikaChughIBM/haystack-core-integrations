# DB2 Haystack Integration - Architecture Documentation

## Overview

The DB2 Haystack integration provides a complete document store implementation for IBM DB2 with native vector support, enabling semantic search, hybrid retrieval, and RAG (Retrieval-Augmented Generation) applications.

**Version**: Beta (Development Status: 4)  
**License**: Apache 2.0  
**Python Support**: 3.10, 3.11, 3.12, 3.13

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Database Schema](#database-schema)
5. [Query Execution](#query-execution)
6. [Filter System](#filter-system)
7. [Retrieval Strategies](#retrieval-strategies)
8. [Examples & Use Cases](#examples--use-cases)
9. [Testing Strategy](#testing-strategy)
10. [Configuration](#configuration)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Haystack Pipeline                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Embedders   │  │  Retrievers  │  │  Generators  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘          │
└─────────┼──────────────────┼──────────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│              DB2 Haystack Integration Layer                      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              DB2DocumentStore (Core)                    │    │
│  │  • Connection Management (thread-local)                 │    │
│  │  • Document CRUD Operations                             │    │
│  │  • Embedding Model Validation                           │    │
│  │  • Metadata Management                                  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Embedding   │  │   Keyword    │  │    Hybrid    │         │
│  │  Retriever   │  │  Retriever   │  │  Retriever   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                 │
│                            │                                     │
│  ┌─────────────────────────▼──────────────────────────┐        │
│  │           Query Builder & Filter System             │        │
│  │  • SQL Generation                                   │        │
│  │  • Filter Translation (Haystack → DB2 SQL)          │        │
│  │  • Parameter Binding                                │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              Converters & Utilities                  │        │
│  │  • Document ↔ DB2 Row Conversion                    │        │
│  │  • Type Mapping                                      │        │
│  │  • Score Calculation (distance → similarity)         │        │
│  └─────────────────────────────────────────────────────┘        │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      IBM DB2 Database                            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Main Document Table                        │    │
│  │  • id (VARCHAR PRIMARY KEY)                             │    │
│  │  • content (CLOB) - Document text                       │    │
│  │  • embedding (VECTOR) - 384/768/1536 dimensions         │    │
│  │  • meta (CLOB) - JSON metadata                          │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │           Metadata Validation Table                     │    │
│  │  • table_name (VARCHAR PRIMARY KEY)                     │    │
│  │  • embedding_model (VARCHAR)                            │    │
│  │  • embedding_dimension (INTEGER)                        │    │
│  │  • distance_metric (VARCHAR)                            │    │
│  │  • created_at (TIMESTAMP)                               │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Native Vector Operations                   │    │
│  │  • VECTOR_DISTANCE(v1, v2, metric)                      │    │
│  │  • Cosine / Euclidean / Inner Product                   │    │
│  │  • JSON_VALUE() for metadata filtering                  │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Document Store (`document_store.py`)

**Location**: `src/haystack_integrations/document_stores/db2/document_store.py`

**Responsibilities**:
- Database connection management (thread-local connections)
- Document lifecycle (CRUD operations)
- Embedding model validation and metadata storage
- Vector and keyword search execution
- Filter application
- Pagination support

**Key Methods**:
```python
# Document Operations
write_documents(documents, policy)
filter_documents(filters)
delete_documents(document_ids)
update_documents(documents)
count_documents()

# Search Operations
query_by_embedding(query_embedding, filters, top_k, offset)
query_by_keyword(query, filters, top_k)

# Metadata Operations
get_metadata_fields_info()
get_metadata_field_unique_values(field)
get_metadata_field_min_max(field)

# Internal Methods
_embedding_retrieval(query_embedding, top_k, filters, return_embedding, offset)
_keyword_retrieval(query, top_k, filters)
```

**Connection Management**:
- Thread-local storage for connections
- Automatic connection pooling per thread
- Connection string or parameter-based configuration
- Support for local and remote DB2 instances

**Embedding Model Validation**:
- Stores model metadata in separate table
- Validates model consistency on initialization
- Prevents model mismatch errors
- Tracks: model name, dimension, distance metric, creation timestamp

---

### 2. Query Builder (`query_builder.py`)

**Location**: `src/haystack_integrations/document_stores/db2/query_builder.py`

**Responsibilities**:
- SQL query generation for all operations
- Vector search query construction
- Keyword search query construction
- Filter clause generation
- Pagination support (OFFSET/FETCH)

**Key Methods**:
```python
build_vector_search(distance_metric, embedding_str, top_k, where_clause, offset)
build_keyword_search(keywords, top_k, where_clause)
build_insert_document(document_dict)
build_update_document(document_id, document_dict)
build_delete_document(document_id)
build_count_query(where_clause)
```

**SQL Generation Examples**:

**Vector Search**:
```sql
SELECT id, content, embedding, meta,
       VECTOR_DISTANCE(
            embedding,
            CAST('[0.1, 0.2, ...]' AS VECTOR(384, FLOAT32)),
            COSINE
        ) as distance
FROM product_vectors
WHERE JSON_VALUE(meta, '$.brand') = 'Nike'
ORDER BY distance ASC 
FETCH FIRST 10 ROWS ONLY
```

**Keyword Search**:
```sql
SELECT id, content, meta,
       (CASE WHEN LOWER(content) LIKE '%keyword1%' THEN 1 ELSE 0 END +
        CASE WHEN LOWER(content) LIKE '%keyword2%' THEN 1 ELSE 0 END) as score
FROM product_vectors
WHERE LOWER(content) LIKE '%keyword1%'
  AND LOWER(content) LIKE '%keyword2%'
ORDER BY score DESC
FETCH FIRST 10 ROWS ONLY
```

---

### 3. Filter System (`filters.py`)

**Location**: `src/haystack_integrations/document_stores/db2/filters.py`

**Responsibilities**:
- Convert Haystack filters to DB2 SQL WHERE clauses
- Handle comparison operators ($eq, $ne, $lt, $gt, $lte, $gte, $in, $nin)
- Handle logical operators ($and, $or, $not)
- Type casting for numeric comparisons
- Parameter binding for SQL injection prevention

**Filter Translation**:

**Input (Haystack Filter)**:
```python
{
    "brand": "Nike",
    "price": {"$lt": 150},
    "in_stock": True
}
```

**Output (SQL WHERE Clause)**:
```sql
WHERE (JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?)
  AND (CAST(CAST(JSON_VALUE(meta, '$.price' RETURNING VARCHAR(1000)) AS VARCHAR(100)) AS DECFLOAT) < CAST(? AS DECFLOAT))
  AND (JSON_VALUE(meta, '$.in_stock' RETURNING VARCHAR(1000)) = ?)
```

**Parameters**: `['Nike', 150, 'true']`

**Supported Operators**:
- **Comparison**: `$eq`, `$ne`, `$lt`, `$gt`, `$lte`, `$gte`
- **Membership**: `$in`, `$nin`
- **Logical**: `$and`, `$or`, `$not`

**Type Handling**:
- **Strings**: Direct JSON_VALUE comparison
- **Numbers**: CAST to DECFLOAT for numeric operations
- **Booleans**: String comparison ("true"/"false")
- **Arrays**: IN/NOT IN clauses

---

### 4. Converters (`converters.py`)

**Location**: `src/haystack_integrations/document_stores/db2/converters.py`

**Responsibilities**:
- Convert Haystack Documents to DB2 row dictionaries
- Convert DB2 rows to Haystack Documents
- Handle embedding serialization/deserialization
- Metadata JSON encoding/decoding
- Score calculation (distance → similarity)

**Key Functions**:
```python
document_to_db2_dict(document) -> dict
db2_row_to_document(row, include_embedding=False) -> Document
```

**Score Conversion**:
```python
# Cosine distance (0-2) → Similarity score (-1 to 1)
score = 1.0 - distance

# Examples:
# distance=0.0 (identical) → score=1.0
# distance=1.0 (orthogonal) → score=0.0
# distance=2.0 (opposite) → score=-1.0
```

---

### 5. Retrievers

#### Embedding Retriever (`embedding_retriever.py`)

**Location**: `src/haystack_integrations/components/retrievers/db2/embedding_retriever.py`

**Purpose**: Semantic search using vector similarity

**Pipeline Integration**:
```python
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", DB2EmbeddingRetriever(document_store))
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
```

**Parameters**:
- `document_store`: DB2DocumentStore instance
- `filters`: Optional metadata filters
- `top_k`: Number of results (default: 10)
- `return_embedding`: Include embeddings in results (default: False)
- `offset`: Pagination offset (default: 0)

#### Keyword Retriever (`keyword_retriever.py`)

**Location**: `src/haystack_integrations/components/retrievers/db2/keyword_retriever.py`

**Purpose**: Keyword-based search using SQL LIKE

**Features**:
- Multi-word keyword matching
- Case-insensitive search
- Score based on keyword match count
- All keywords must be present (AND logic)

#### Hybrid Retriever (`hybrid_retriever.py`)

**Location**: `src/haystack_integrations/components/retrievers/db2/hybrid_retriever.py`

**Purpose**: Combine vector and keyword search with RRF (Reciprocal Rank Fusion)

**Algorithm**:
```python
# RRF Score Calculation
rrf_score = Σ(1 / (k + rank_i))

# Where:
# k = 60 (constant)
# rank_i = position in result list (1-based)
```

**Pipeline Integration**:
```python
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", DB2HybridRetriever(document_store))
pipeline.add_component("joiner", DocumentJoiner())

pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "joiner.documents")
```

---

## Data Flow

### Indexing Pipeline

```
┌──────────────┐
│   Raw Data   │
│  (JSON/CSV)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Load Documents      │
│  (Python objects)    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Document Embedder   │
│  (SentenceTransf.)   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Document Writer     │
│  (Haystack)          │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  DB2DocumentStore    │
│  write_documents()   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Query Builder       │
│  build_insert()      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  DB2 Database        │
│  INSERT INTO table   │
└──────────────────────┘
```

### Retrieval Pipeline (Vector Search)

```
┌──────────────┐
│  User Query  │
│  "running    │
│   shoes"     │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Text Embedder       │
│  (SentenceTransf.)   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Embedding Retriever │
│  (DB2)               │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  DB2DocumentStore    │
│  query_by_embedding()│
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Query Builder       │
│  build_vector_search │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  DB2 Database        │
│  VECTOR_DISTANCE()   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Converters          │
│  db2_row_to_document │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Ranked Documents    │
│  (with scores)       │
└──────────────────────┘
```

### Hybrid Retrieval Pipeline

```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │
       ├─────────────────────┬─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Embedder  │      │   Keyword   │      │   Filters   │
│             │      │   Parser    │      │   Parser    │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │ Hybrid Retriever │
                   └────────┬─────────┘
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Vector    │      │   Keyword   │      │   Filter    │
│   Search    │      │   Search    │      │   Apply     │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │  RRF Fusion      │
                   │  (Rank Merging)  │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Document Joiner  │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Final Results    │
                   └──────────────────┘
```

---

## Database Schema

### Main Document Table

```sql
CREATE TABLE product_vectors (
    id VARCHAR(255) NOT NULL PRIMARY KEY,
    content CLOB,
    embedding VECTOR(384, FLOAT32),
    meta CLOB
)
```

**Columns**:
- `id`: Unique document identifier (VARCHAR PRIMARY KEY)
- `content`: Document text content (CLOB for large text)
- `embedding`: Vector representation (VECTOR type, dimension configurable)
- `meta`: JSON metadata (CLOB storing JSON string)

### Metadata Validation Table

```sql
CREATE TABLE product_vectors_metadata (
    table_name VARCHAR(255) NOT NULL PRIMARY KEY,
    embedding_model VARCHAR(500),
    embedding_dimension INTEGER,
    distance_metric VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Purpose**: Prevent embedding model mismatch errors

**Validation Logic**:
1. On first write: Store model metadata
2. On subsequent writes: Validate against stored metadata
3. On mismatch: Raise error with clear message

---

## Query Execution

### Vector Search Query

**SQL Template**:
```sql
SELECT id, content, embedding, meta,
       VECTOR_DISTANCE(
            embedding,
            CAST(? AS VECTOR(384, FLOAT32)),
            COSINE
        ) as distance
FROM {table_name}
{where_clause}
ORDER BY distance ASC
OFFSET {offset} ROWS
FETCH FIRST {top_k} ROWS ONLY
```

**Parameters**:
- `embedding`: Query vector as string "[0.1, 0.2, ...]"
- `where_clause`: Optional filter conditions
- `offset`: Pagination offset
- `top_k`: Number of results

**Distance Metrics**:
- `COSINE`: Cosine distance (0-2, lower is better)
- `EUCLIDEAN`: Euclidean distance
- `INNER_PRODUCT`: Inner product (dot product)

### Keyword Search Query

**SQL Template**:
```sql
SELECT id, content, meta,
       (CASE WHEN LOWER(content) LIKE ? THEN 1 ELSE 0 END +
        CASE WHEN LOWER(content) LIKE ? THEN 1 ELSE 0 END +
        ...) as score
FROM {table_name}
WHERE LOWER(content) LIKE ?
  AND LOWER(content) LIKE ?
  AND ...
{additional_filters}
ORDER BY score DESC
FETCH FIRST {top_k} ROWS ONLY
```

**Features**:
- Case-insensitive matching
- Score = number of matching keywords
- All keywords must be present (AND logic)

---

## Filter System

### Filter Translation Examples

#### Simple Equality
```python
# Input
{"brand": "Nike"}

# SQL
WHERE JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?
# Params: ['Nike']
```

#### Numeric Comparison
```python
# Input
{"price": {"$lt": 150}}

# SQL
WHERE CAST(CAST(JSON_VALUE(meta, '$.price' RETURNING VARCHAR(1000)) AS VARCHAR(100)) AS DECFLOAT) < CAST(? AS DECFLOAT)
# Params: [150]
```

#### Multiple Conditions (AND)
```python
# Input
{
    "brand": "Nike",
    "price": {"$lt": 150},
    "in_stock": True
}

# SQL
WHERE (JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?)
  AND (CAST(CAST(JSON_VALUE(meta, '$.price' RETURNING VARCHAR(1000)) AS VARCHAR(100)) AS DECFLOAT) < CAST(? AS DECFLOAT))
  AND (JSON_VALUE(meta, '$.in_stock' RETURNING VARCHAR(1000)) = ?)
# Params: ['Nike', 150, 'true']
```

#### IN Operator
```python
# Input
{"brand": {"$in": ["Nike", "Adidas"]}}

# SQL
WHERE JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) IN (?, ?)
# Params: ['Nike', 'Adidas']
```

#### Logical OR
```python
# Input
{
    "$or": [
        {"brand": "Nike"},
        {"brand": "Adidas"}
    ]
}

# SQL
WHERE (JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?)
   OR (JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?)
# Params: ['Nike', 'Adidas']
```

---

## Retrieval Strategies

### 1. Pure Vector Search

**Use Case**: Semantic similarity search

**Example**:
```python
results = document_store.query_by_embedding(
    query_embedding=[0.1, 0.2, ...],
    top_k=10
)
```

**Advantages**:
- Understands semantic meaning
- Works with synonyms and paraphrases
- Language-agnostic (with multilingual models)

**Limitations**:
- Requires embedding model
- Computationally expensive
- May miss exact keyword matches

### 2. Keyword Search

**Use Case**: Exact term matching

**Example**:
```python
results = document_store.query_by_keyword(
    query="Nike running shoes",
    top_k=10
)
```

**Advantages**:
- Fast execution
- No embedding required
- Exact term matching

**Limitations**:
- No semantic understanding
- Misses synonyms
- Case-sensitive (mitigated with LOWER())

### 3. Hybrid Search (Vector + Keyword + RRF)

**Use Case**: Best of both worlds

**Example**:
```python
pipeline = Pipeline()
pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", DB2HybridRetriever(document_store))
pipeline.add_component("joiner", DocumentJoiner())

results = pipeline.run({
    "embedder": {"text": "Nike running shoes"},
    "retriever": {"query": "Nike running shoes"}
})
```

**RRF Formula**:
```
score(doc) = Σ(1 / (k + rank_i))

Where:
- k = 60 (constant)
- rank_i = position in result list from retriever i
```

**Advantages**:
- Combines semantic and keyword matching
- Robust to different query types
- Better recall and precision

### 4. Vector Search with SQL Filters

**Use Case**: Semantic search with structured constraints

**Example**:
```python
results = document_store.query_by_embedding(
    query_embedding=[0.1, 0.2, ...],
    filters={
        "brand": "Nike",
        "price": {"$lt": 150},
        "in_stock": True
    },
    top_k=10
)
```

**Advantages**:
- Pre-filters before vector search
- Reduces search space
- Enforces business rules

**SQL Execution**:
```sql
-- Filters applied BEFORE vector distance calculation
WHERE (brand = 'Nike') AND (price < 150) AND (in_stock = true)
ORDER BY VECTOR_DISTANCE(...) ASC
```

---

## Examples & Use Cases

### Basic Usage (`basic_usage.py`)
- Simple document indexing
- Basic vector search
- Document CRUD operations

### Embedding Retrieval (`embedding_retrieval.py`)
- Pure semantic search
- Embedding model integration
- Score interpretation

### Hybrid Retrieval (`hybrid_retrieval.py`)
- Vector + Keyword fusion
- RRF scoring
- Document joiner usage

### Product Search (`product_search_hybrid.py`)
- Real-world e-commerce scenario
- Intelligent query parsing
- Filter extraction from natural language
- 18 search scenarios:
  - Pure semantic search (6 scenarios)
  - Hybrid search with filters (4 scenarios)
  - Intelligent parsing (8 scenarios)

### Model Validation (`model_validation_example.py`)
- Embedding model validation
- Metadata table usage
- Error prevention

### Reranking (`reranking_example.py`)
- Post-retrieval reranking
- Cross-encoder integration
- Score refinement

---

## Testing Strategy

### Unit Tests
- `test_converters.py`: Document ↔ DB2 conversion
- `test_filters.py`: Filter translation logic
- `test_score_calculation.py`: Distance → similarity conversion

### Integration Tests
- `test_document_store.py`: Full CRUD operations
- `test_retrieval.py`: Vector and keyword search
- `test_hybrid_retriever.py`: RRF fusion
- `test_pagination.py`: Offset/limit functionality
- `test_filter_policy.py`: Filter application
- `test_schema_support.py`: Table creation/validation

### Real-World Tests
- `test_real_world_scenarios.py`: E-commerce use cases
- `test_reranking.py`: Reranking pipeline
- `test_db2_haystack_integration.py`: End-to-end integration

---

## Configuration

### Environment Variables

```bash
# Local DB2 Connection
DB2_USER=your_username
DB2_PASSWORD=your_password
DB2_DATABASE=TESTDB
DB2_HOST=localhost
DB2_PORT=50000

# Remote DB2 Connection
DB2_CONNECTION_STRING=DATABASE=BLUDB;HOSTNAME=host.cloud;PORT=32310;...
```

### Document Store Initialization

```python
from haystack_integrations.document_stores.db2 import DB2DocumentStore
from haystack.utils import Secret

# Method 1: Individual parameters
document_store = DB2DocumentStore(
    database="TESTDB",
    username=Secret.from_env_var("DB2_USER"),
    password=Secret.from_env_var("DB2_PASSWORD"),
    hostname="localhost",
    port=50000,
    table_name="documents",
    embedding_dimension=384,
    distance_metric="cosine",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    validate_embedding_model=True,
    recreate_table=False,
    batch_size=100
)

# Method 2: Connection string
document_store = DB2DocumentStore(
    connection_string=Secret.from_env_var("DB2_CONNECTION_STRING"),
    table_name="documents",
    embedding_dimension=384
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `database` | str | Required | Database name |
| `username` | Secret | Required | DB2 username |
| `password` | Secret | Required | DB2 password |
| `hostname` | str | "localhost" | DB2 hostname |
| `port` | int | 50000 | DB2 port |
| `connection_string` | Secret | None | Full connection string (alternative) |
| `table_name` | str | "haystack_documents" | Table name |
| `embedding_dimension` | int | 768 | Vector dimension |
| `distance_metric` | str | "cosine" | Distance metric (cosine/euclidean/inner_product) |
| `embedding_model` | str | None | Model name for validation |
| `validate_embedding_model` | bool | False | Enable model validation |
| `recreate_table` | bool | False | Drop and recreate table |
| `batch_size` | int | 100 | Batch size for writes |

---

## Key Design Decisions

### 1. Thread-Local Connections
**Rationale**: ibm_db is not thread-safe. Thread-local storage ensures each thread has its own connection.

### 2. Score Conversion (1 - distance)
**Rationale**: Haystack expects higher scores = better matches. Cosine distance (0-2) is inverted to similarity (-1 to 1).

### 3. JSON Metadata Storage
**Rationale**: Flexible schema, supports arbitrary metadata fields without ALTER TABLE.

### 4. Embedding Model Validation
**Rationale**: Prevents silent errors from model mismatches. Stores metadata in separate table.

### 5. Double CAST for Numeric Filters
**Rationale**: DB2 JSON_VALUE returns VARCHAR. Must cast to DECFLOAT for numeric comparisons.

### 6. Pagination with OFFSET/FETCH
**Rationale**: Standard SQL pagination. Efficient for large result sets.

### 7. No Async Support
**Rationale**: ibm_db library is synchronous-only. Async methods raise NotImplementedError with clear message.

---

## Performance Considerations

### Indexing
- **Batch Size**: Default 100, configurable
- **Embedding Generation**: Bottleneck for large datasets
- **Connection Pooling**: Thread-local connections reduce overhead

### Retrieval
- **Vector Search**: O(n) complexity, scales with table size
- **Keyword Search**: O(n) with LIKE, consider full-text indexes
- **Filters**: Applied before vector distance calculation (reduces search space)

### Optimization Tips
1. Use filters to reduce search space
2. Adjust `top_k` based on needs (lower = faster)
3. Consider pagination for large result sets
4. Use appropriate distance metric (cosine for normalized vectors)
5. Batch document writes for better throughput

---

## Future Enhancements

### Planned Features
- [ ] Async support (pending ibm_db library update)
- [ ] Full-text search indexes
- [ ] Approximate nearest neighbor (ANN) indexes
- [ ] Multi-vector support
- [ ] Sparse vector support
- [ ] Query caching
- [ ] Connection pooling improvements

### Under Consideration
- [ ] Distributed deployment support
- [ ] Sharding strategies
- [ ] Backup/restore utilities
- [ ] Migration tools
- [ ] Performance monitoring

---

## Troubleshooting

### Common Issues

**1. Connection Errors**
```
Error: SQL30081N  A communication error has been detected
```
**Solution**: Check hostname, port, firewall settings

**2. Model Mismatch**
```
Error: Embedding model mismatch. Expected 'model-a', got 'model-b'
```
**Solution**: Use same model or set `validate_embedding_model=False`

**3. Dimension Mismatch**
```
Error: Query embedding dimension 768 doesn't match store dimension 384
```
**Solution**: Ensure query embedding matches store dimension

**4. Filter Errors**
```
Error: Invalid filter format
```
**Solution**: Check filter syntax, use supported operators

**5. Type Errors in Filters**
```
Error: Cannot compare string to number
```
**Solution**: Ensure filter values match metadata types

---

## References

- [Haystack Documentation](https://docs.haystack.deepset.ai/)
- [IBM DB2 Vector Documentation](https://www.ibm.com/docs/en/db2)
- [ibm_db Python Driver](https://github.com/ibmdb/python-ibmdb)
- [Sentence Transformers](https://www.sbert.net/)

---

**Last Updated**: 2026-05-10  
**Version**: 1.0.0  
**Maintainer**: deepset GmbH