#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate a single test query for manual DB2 CLI testing.

Usage:
    python3 generate_test_query.py "your query text here"
    
Example:
    python3 generate_test_query.py "running shoes"
"""

import sys
from sentence_transformers import SentenceTransformer

if len(sys.argv) < 2:
    print("Usage: python3 generate_test_query.py 'your query text'")
    print("\nExamples:")
    print("  python3 generate_test_query.py 'shoes'")
    print("  python3 generate_test_query.py 'comfortable running shoes'")
    print("  python3 generate_test_query.py 'professional racing shoe with carbon fiber'")
    sys.exit(1)

query_text = " ".join(sys.argv[1:])

print(f"Generating embedding for query: '{query_text}'")
print("Loading model...")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding = model.encode(query_text).tolist()
embedding_str = "[" + ",".join(map(str, embedding)) + "]"

print(f"Embedding generated: {len(embedding)} dimensions")
print(f"\nCopy the SQL query below and paste into DB2 CLI:\n")
print("=" * 80)

sql = f"SELECT id, SUBSTR(content, 1, 80) as content_preview, VECTOR_DISTANCE(embedding, CAST('{embedding_str}' AS VECTOR(384, FLOAT32)), COSINE) as distance, DECIMAL((1.0 - VECTOR_DISTANCE(embedding, CAST('{embedding_str}' AS VECTOR(384, FLOAT32)), COSINE)), 5, 4) as score FROM product_vectors ORDER BY distance ASC FETCH FIRST 5 ROWS ONLY;"

print(sql)
print("=" * 80)
print(f"\nQuery length: {len(sql)} characters")
print("\nTo run in DB2 CLI:")
print('  db2 "' + sql + '"')

# Made with Bob
