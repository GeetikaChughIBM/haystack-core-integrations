# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Embedding Model Validation Example

This example demonstrates the embedding model validation feature that prevents
using incompatible embedding models for indexing and retrieval.

Key Points:
1. The document store tracks which embedding model was used for indexing
2. On subsequent connections, it validates that the same model is being used
3. Using a different model (even with same dimension) will raise an error
4. This prevents silent failures where search results would be meaningless

Requirements:
- DB2 database with vector support
- Sentence Transformers
- Haystack framework
"""

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore

print("=" * 80)
print("EMBEDDING MODEL VALIDATION EXAMPLE")
print("=" * 80)

# Sample documents
documents = [
    Document(content="Python is a high-level programming language"),
    Document(content="Machine learning is a subset of artificial intelligence"),
    Document(content="Neural networks are inspired by biological neurons"),
]

# ============================================================================
# SCENARIO 1: First Time Setup - Model is Stored
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO 1: First Time Setup")
print("=" * 80)

print("\nCreating document store with embedding model specified...")
document_store = DB2DocumentStore(
    database="TESTDB",
    username=Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1"),
    password=Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password"),
    table_name="model_validation_demo",
    embedding_dimension=384,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # IMPORTANT: Specify model
    recreate_table=True,
)

print("✓ Document store created")
print(f"  Embedding model: {document_store.embedding_model}")
print(f"  Embedding dimension: {document_store.embedding_dimension}")

# Index documents with the specified model
print("\nIndexing documents with all-MiniLM-L6-v2...")
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
)
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
indexing_pipeline.connect("embedder", "writer")

indexing_pipeline.run({"embedder": {"documents": documents}})
print("✓ Documents indexed successfully")
print("  Model information stored in database metadata")

# ============================================================================
# SCENARIO 2: Correct Usage - Same Model
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO 2: Correct Usage - Same Model")
print("=" * 80)

print("\nReconnecting with the SAME embedding model...")
document_store_correct = DB2DocumentStore(
    database="TESTDB",
    username=Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1"),
    password=Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password"),
    table_name="model_validation_demo",
    embedding_dimension=384,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # SAME MODEL ✓
)

print("✓ Connection successful - model matches!")
print(f"  Stored model: sentence-transformers/all-MiniLM-L6-v2")
print(f"  Current model: {document_store_correct.embedding_model}")

# Search works correctly
print("\nPerforming search with matching model...")
query_pipeline = Pipeline()
query_pipeline.add_component(
    "embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
)
query_pipeline.add_component("retriever", DB2EmbeddingRetriever(document_store=document_store_correct, top_k=2))
query_pipeline.connect("embedder.embedding", "retriever.query_embedding")

results = query_pipeline.run({"embedder": {"text": "What is Python?"}})
print(f"✓ Search successful - found {len(results['retriever']['documents'])} results")
for i, doc in enumerate(results["retriever"]["documents"], 1):
    print(f"  {i}. {doc.content[:50]}... (score: {doc.score:.4f})")

# ============================================================================
# SCENARIO 3: Wrong Model - Validation Error
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO 3: Wrong Model - Validation Error")
print("=" * 80)

print("\nAttempting to connect with a DIFFERENT embedding model...")
print("  Stored model: sentence-transformers/all-MiniLM-L6-v2")
print("  Trying model: sentence-transformers/paraphrase-MiniLM-L6-v2")
print("  (Both have 384 dimensions, but incompatible embedding spaces!)")

try:
    document_store_wrong = DB2DocumentStore(
        database="TESTDB",
        username=Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1"),
        password=Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password"),
        table_name="model_validation_demo",
        embedding_dimension=384,
        embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2",  # DIFFERENT MODEL ✗
    )
    print("✗ ERROR: Should have raised ValueError!")
except ValueError as e:
    print("✓ Validation error caught successfully!")
    print("\nError message:")
    print(str(e))

# ============================================================================
# SCENARIO 4: Bypass Validation (Not Recommended)
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO 4: Bypass Validation (Not Recommended)")
print("=" * 80)

print("\nConnecting with validation disabled...")
print("  WARNING: This bypasses safety checks!")

document_store_bypass = DB2DocumentStore(
    database="TESTDB",
    username=Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1"),
    password=Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password"),
    table_name="model_validation_demo",
    embedding_dimension=384,
    embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2",
    validate_embedding_model=False,  # BYPASS VALIDATION
)

print("✓ Connection successful (validation bypassed)")
print("  ⚠️  Search results will be MEANINGLESS with wrong model!")

# ============================================================================
# SCENARIO 5: No Model Specified (Backward Compatible)
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO 5: No Model Specified (Backward Compatible)")
print("=" * 80)

print("\nConnecting without specifying embedding_model...")
document_store_no_model = DB2DocumentStore(
    database="TESTDB",
    username=Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1"),
    password=Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password"),
    table_name="model_validation_demo",
    embedding_dimension=384,
    # embedding_model not specified
)

print("✓ Connection successful (no validation performed)")
print("  Note: Validation only occurs when embedding_model is specified")
print("  Recommendation: Always specify embedding_model for production use")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
✓ BEST PRACTICES:

1. Always specify embedding_model parameter:
   document_store = DB2DocumentStore(
       ...,
       embedding_model="sentence-transformers/all-MiniLM-L6-v2"
   )

2. Use the SAME model for indexing and retrieval:
   - Indexing: SentenceTransformersDocumentEmbedder(model="all-MiniLM-L6-v2")
   - Retrieval: SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2")

3. Keep validate_embedding_model=True (default):
   - Prevents accidental model mismatches
   - Catches errors early
   - Ensures search quality

⚠️  COMMON MISTAKES TO AVOID:

1. Using different models with same dimension:
   ✗ Index with: all-MiniLM-L6-v2 (384 dims)
   ✗ Query with: paraphrase-MiniLM-L6-v2 (384 dims)
   → Dimensions match but embeddings are INCOMPATIBLE!

2. Bypassing validation without understanding consequences:
   ✗ validate_embedding_model=False
   → Search results will be meaningless

3. Not specifying embedding_model:
   ✗ embedding_model=None
   → No protection against model mismatches

🎯 REMEMBER:
   Same dimension ≠ Compatible embeddings
   Always use the SAME model for indexing and retrieval!
""")

print("=" * 80)
print("Example complete!")
print("=" * 80)

# Made with Bob
