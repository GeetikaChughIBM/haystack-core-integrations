"""
Complete RAG pipeline example using DB2DocumentStore.

This example demonstrates:
1. Setting up a DB2 document store
2. Creating an indexing pipeline with embeddings
3. Creating a query pipeline with retrieval
4. Running end-to-end semantic search
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Auto-load credentials from .env file in the db2 integration directory
try:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"✓ Loaded credentials from {env_path}\n")
    else:
        # Fallback to current directory
        load_dotenv()
except ImportError:
    logger.info("⚠ python-dotenv not installed\n")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Initialize DB2 document store
use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
port = int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000"))

document_store = DB2DocumentStore(
    database=os.getenv("DB2_DATABASE", "TESTDB"),
    hostname=os.getenv("DB2_HOSTNAME"),
    port=port,
    username=Secret.from_env_var("DB2_USER"),
    password=Secret.from_env_var("DB2_PASSWORD"),
    table_name="haystack_rag_demo",
    embedding_dimension=384,  # all-MiniLM-L6-v2 dimension
    distance_metric="cosine",
    use_ssl=use_ssl,
    ssl_certificate=os.getenv("DB2_SSL_CERTIFICATE") or os.getenv("DB2_SSL_CERT_PATH"),
    recreate_table=True,
)

logger.info("%s", "=" * 80)
logger.info("DB2 RAG Pipeline Example")
logger.info("%s", "=" * 80)

# Step 1: Create indexing pipeline
logger.info("\n1. Creating indexing pipeline...")
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(
    "embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
indexing_pipeline.connect("embedder", "writer")

# Step 2: Prepare documents
logger.info("\n2. Preparing documents...")
documents = [
    Document(
        content="Machine learning is a subset of artificial intelligence that focuses on "
        "enabling computers to learn from data without being explicitly programmed.",
        meta={"category": "AI", "source": "textbook", "priority": 1},
    ),
    Document(
        content="Python is a high-level programming language known for its simplicity and "
        "readability, making it popular for data science and machine learning.",
        meta={"category": "Programming", "source": "tutorial", "priority": 2},
    ),
    Document(
        content="Deep learning is a subset of machine learning that uses neural networks "
        "with multiple layers to model complex patterns in data.",
        meta={"category": "AI", "source": "research", "priority": 1},
    ),
    Document(
        content="Natural language processing (NLP) is a field of AI that focuses on the "
        "interaction between computers and human language.",
        meta={"category": "AI", "source": "textbook", "priority": 2},
    ),
    Document(
        content="Data science combines statistics, programming, and domain expertise to extract insights from data.",
        meta={"category": "Data Science", "source": "tutorial", "priority": 3},
    ),
]

# Step 3: Index documents
logger.info("\n3. Indexing documents...")
indexing_pipeline.run({"embedder": {"documents": documents}})
logger.info(f"   ✓ Indexed {document_store.count_documents()} documents")

# Step 4: Create query pipeline
logger.info("\n4. Creating query pipeline...")
query_pipeline = Pipeline()
query_pipeline.add_component(
    "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
query_pipeline.add_component("retriever", DB2EmbeddingRetriever(document_store=document_store, top_k=3))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

# Step 5: Run queries
logger.info("\n5. Running queries...")
logger.info("%s", "\n" + "-" * 80)

queries = [
    "What is machine learning?",
    "Tell me about Python programming",
    "How does deep learning work?",
]

for query in queries:
    logger.info(f"\nQuery: {query}")
    logger.info("%s", "-" * 40)

    results = query_pipeline.run({"text_embedder": {"text": query}})
    documents_found = results["retriever"]["documents"]

    for i, doc in enumerate(documents_found, 1):
        score = doc.score if doc.score is not None else 0.0
        category = doc.meta.get("category", "Unknown")
        logger.info(f"\n  Result {i} (score: {score:.4f}, category: {category}):")
        logger.info(f"  {doc.content[:100]}...")

logger.info("%s", "\n" + "=" * 80)

# Step 6: Demonstrate filtering
logger.info("\n6. Demonstrating filtered retrieval...")
logger.info("%s", "-" * 80)

# Create retriever with filter
filtered_retriever = DB2EmbeddingRetriever(
    document_store=document_store, filters={"field": "category", "operator": "==", "value": "AI"}, top_k=5
)

filtered_pipeline = Pipeline()
filtered_pipeline.add_component(
    "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
filtered_pipeline.add_component("retriever", filtered_retriever)
filtered_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "Tell me about artificial intelligence"
logger.info(f"\nQuery: {query}")
logger.info("Filter: category = 'AI'")
logger.info("%s", "-" * 40)

results = filtered_pipeline.run({"text_embedder": {"text": query}})
documents_found = results["retriever"]["documents"]

logger.info(f"\nFound {len(documents_found)} AI-related documents:")
for i, doc in enumerate(documents_found, 1):
    score = doc.score if doc.score is not None else 0.0
    logger.info(f"\n  Result {i} (score: {score:.4f}):")
    logger.info(f"  {doc.content[:100]}...")

logger.info("%s", "\n" + "=" * 80)

# Step 7: Demonstrate pagination in query_by_embedding
logger.info("\n7. Demonstrating pagination with query_by_embedding...")
logger.info("%s", "-" * 80)

# Get query embedding
from haystack.components.embedders import SentenceTransformersTextEmbedder

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
text_embedder.warm_up()

query = "What is artificial intelligence?"
embedding_result = text_embedder.run(text=query)
query_embedding = embedding_result["embedding"]

logger.info(f"\nQuery: {query}")
logger.info("Demonstrating pagination with offset parameter")
logger.info("%s", "-" * 40)

# Get first page (top 2 results)
logger.info("\nPage 1 (offset=0, top_k=2):")
page1_results = document_store.query_by_embedding(query_embedding=query_embedding, top_k=2, offset=0)
for i, doc in enumerate(page1_results, 1):
    logger.info(f"  {i}. Score: {doc.score:.4f} - {str(doc.content)[:60]}...")

# Get second page (next 2 results)
logger.info("\nPage 2 (offset=2, top_k=2):")
page2_results = document_store.query_by_embedding(query_embedding=query_embedding, top_k=2, offset=2)
for i, doc in enumerate(page2_results, 1):
    logger.info(f"  {i}. Score: {doc.score:.4f} - {str(doc.content)[:60]}...")

# Get third page (remaining results)
logger.info("\nPage 3 (offset=4, top_k=2):")
page3_results = document_store.query_by_embedding(query_embedding=query_embedding, top_k=2, offset=4)
for i, doc in enumerate(page3_results, 1):
    logger.info(f"  {i}. Score: {doc.score:.4f} - {str(doc.content)[:60]}...")

logger.info("\n✓ Pagination allows you to retrieve results in pages for better UX")
logger.info("✓ Note: Scores use formula score = 1 - distance (cosine) where higher = better match")

logger.info("%s", "\n" + "=" * 80)
logger.info("✓ RAG pipeline example completed successfully!")
logger.info("%s", "=" * 80)
