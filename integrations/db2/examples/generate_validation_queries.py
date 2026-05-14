#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Manual validation script - generates complete SQL queries for DB2 vector search.

This script generates ready-to-execute SQL queries with full embedding vectors.
Each query is on a SINGLE LINE to work with DB2 CLI.

Usage:
    python3 generate_validation_queries.py > validation_queries.sql
    
Then run in DB2:
    db2 connect to TESTDB
    db2 -tvf validation_queries.sql
"""

from sentence_transformers import SentenceTransformer

# Load the SAME model used in the pipeline
print("-- Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("-- Model loaded successfully!")
print("--")

# Query 1: Broad search - should return multiple products
query1 = "shoes"
embedding1 = model.encode(query1).tolist()
embedding1_str = "[" + ",".join(map(str, embedding1)) + "]"

print("-- ============================================================================")
print("-- QUERY 1: Broad Search - 'shoes'")
print("-- Expected: All 6 products (Nike and Adidas shoes)")
print("-- ============================================================================")
# Single-line SQL for DB2 CLI compatibility
sql1 = f"SELECT id, SUBSTR(content, 1, 80) as content_preview, VECTOR_DISTANCE(embedding, CAST('{embedding1_str}' AS VECTOR(384, FLOAT32)), COSINE) as distance, DECIMAL((1.0 - VECTOR_DISTANCE(embedding, CAST('{embedding1_str}' AS VECTOR(384, FLOAT32)), COSINE)), 5, 4) as score FROM product_vectors ORDER BY distance ASC FETCH FIRST 6 ROWS ONLY;"
print(sql1)
print("--")

# Query 2: Specific search - should return 1-2 most relevant products
query2 = "professional racing shoe with carbon fiber plate for competitive runners"
embedding2 = model.encode(query2).tolist()
embedding2_str = "[" + ",".join(map(str, embedding2)) + "]"

print("-- ============================================================================")
print("-- QUERY 2: Specific Search - Racing shoe with carbon fiber")
print("-- Expected: nike_003 (ZoomX Vaporfly) with highest score")
print("-- ============================================================================")
sql2 = f"SELECT id, SUBSTR(content, 1, 80) as content_preview, VECTOR_DISTANCE(embedding, CAST('{embedding2_str}' AS VECTOR(384, FLOAT32)), COSINE) as distance, DECIMAL((1.0 - VECTOR_DISTANCE(embedding, CAST('{embedding2_str}' AS VECTOR(384, FLOAT32)), COSINE)), 5, 4) as score FROM product_vectors ORDER BY distance ASC FETCH FIRST 2 ROWS ONLY;"
print(sql2)
print("--")

# Query 3: Comfort-focused search
query3 = "comfortable cushioned running shoes for daily training"
embedding3 = model.encode(query3).tolist()
embedding3_str = "[" + ",".join(map(str, embedding3)) + "]"

print("-- ============================================================================")
print("-- QUERY 3: Comfort-Focused Search")
print("-- Expected: adidas_001 (Ultraboost) and nike_001 (Air Max) with high scores")
print("-- ============================================================================")
sql3 = f"SELECT id, SUBSTR(content, 1, 80) as content_preview, VECTOR_DISTANCE(embedding, CAST('{embedding3_str}' AS VECTOR(384, FLOAT32)), COSINE) as distance, DECIMAL((1.0 - VECTOR_DISTANCE(embedding, CAST('{embedding3_str}' AS VECTOR(384, FLOAT32)), COSINE)), 5, 4) as score FROM product_vectors ORDER BY distance ASC FETCH FIRST 3 ROWS ONLY;"
print(sql3)
print("--")

print("-- ============================================================================")
print("-- VALIDATION COMPLETE")
print("-- Compare these results with the Python pipeline output")
print("-- ============================================================================")

# Made with Bob
