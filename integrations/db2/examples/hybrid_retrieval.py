"""
Hybrid retrieval example combining keyword and embedding search.

This example demonstrates:
- Setting up both keyword and embedding retrievers
- Combining results using reciprocal rank fusion
- Comparing keyword vs embedding vs hybrid search results

Prerequisites:
- DB2 database running (v12.1.2+ with vector support)
- Environment variables set (see .env.example)
- Install: pip install db2-haystack sentence-transformers
"""

import logging
import os

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.utils import Secret

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever, DB2KeywordRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run hybrid retrieval example."""

    logger.info("%s", "=" * 70)
    logger.info("DB2 Haystack Integration - Hybrid Retrieval Example")
    logger.info("%s", "=" * 70)

    # Initialize document store
    logger.info("\n1. Initializing DB2 Document Store...")
    document_store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        hostname=os.getenv("DB2_HOST", "localhost"),
        port=int(os.getenv("DB2_PORT", "50000")),
        table_name="hybrid_example",
        embedding_dimension=384,
        distance_metric="cosine",
        recreate_table=True,
    )
    logger.info("✓ Document store initialized")

    # Create sample documents
    logger.info("\n2. Creating sample documents...")
    documents = [
        Document(
            content="Python is a high-level programming language known for its simplicity and readability.",
            meta={"category": "programming", "language": "Python"},
        ),
        Document(
            content="JavaScript is the programming language of the web, used for frontend and backend development.",
            meta={"category": "programming", "language": "JavaScript"},
        ),
        Document(
            content="Machine learning algorithms can learn patterns from data without explicit programming.",
            meta={"category": "AI", "topic": "machine learning"},
        ),
        Document(
            content="Deep learning uses neural networks with multiple layers to process complex patterns.",
            meta={"category": "AI", "topic": "deep learning"},
        ),
        Document(
            content="Natural language processing enables computers to understand and generate human language.",
            meta={"category": "AI", "topic": "NLP"},
        ),
        Document(
            content="Python libraries like TensorFlow and PyTorch are popular for machine learning projects.",
            meta={"category": "programming", "language": "Python", "topic": "ML libraries"},
        ),
        Document(
            content="React is a JavaScript library for building user interfaces with reusable components.",
            meta={"category": "programming", "language": "JavaScript", "topic": "frontend"},
        ),
        Document(
            content="SQL databases store structured data in tables with rows and columns.",
            meta={"category": "databases", "type": "SQL"},
        ),
        Document(
            content="NoSQL databases like MongoDB provide flexible schema for unstructured data.",
            meta={"category": "databases", "type": "NoSQL"},
        ),
        Document(
            content="Docker containers package applications with their dependencies for consistent deployment.",
            meta={"category": "DevOps", "tool": "Docker"},
        ),
    ]
    logger.info(f"✓ Created {len(documents)} documents")

    # Index documents with embeddings
    logger.info("\n3. Indexing documents with embeddings...")
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        "embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    indexing_pipeline.connect("embedder", "writer")
    indexing_pipeline.run({"embedder": {"documents": documents}})
    logger.info(f"✓ Indexed {document_store.count_documents()} documents")

    # Create hybrid search pipeline
    logger.info("\n4. Setting up hybrid search pipeline...")
    hybrid_pipeline = Pipeline()

    # Add components
    hybrid_pipeline.add_component(
        "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    hybrid_pipeline.add_component("embedding_retriever", DB2EmbeddingRetriever(document_store=document_store, top_k=5))
    hybrid_pipeline.add_component("keyword_retriever", DB2KeywordRetriever(document_store=document_store, top_k=5))
    hybrid_pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion"))

    # Connect components
    hybrid_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    hybrid_pipeline.connect("embedding_retriever.documents", "joiner.documents")
    hybrid_pipeline.connect("keyword_retriever.documents", "joiner.documents")
    logger.info("✓ Hybrid pipeline created")

    # Example 1: Compare keyword vs embedding vs hybrid
    logger.info("\n5. Example 1: Comparing retrieval methods")
    logger.info("%s", "=" * 70)
    query = "Python machine learning"

    # Keyword-only search
    logger.info(f"\nA. Keyword Search for: '{query}'")
    logger.info("%s", "-" * 70)
    keyword_results = DB2KeywordRetriever(document_store=document_store, top_k=3).run(query=query)
    logger.info(f"Found {len(keyword_results['documents'])} results:\n")
    for i, doc in enumerate(keyword_results["documents"], 1):
        logger.info(f"{i}. {doc.content[:80]}...")
        logger.info(f"   Category: {doc.meta.get('category', 'N/A')}\n")

    # Embedding-only search
    logger.info(f"\nB. Embedding Search for: '{query}'")
    logger.info("%s", "-" * 70)
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = text_embedder.run(query)["embedding"]
    embedding_results = DB2EmbeddingRetriever(document_store=document_store, top_k=3).run(
        query_embedding=query_embedding
    )
    logger.info(f"Found {len(embedding_results['documents'])} results:\n")
    for i, doc in enumerate(embedding_results["documents"], 1):
        score = doc.score if doc.score is not None else 0.0
        logger.info(f"{i}. Score: {score:.4f}")
        logger.info(f"   {doc.content[:80]}...")
        logger.info(f"   Category: {doc.meta.get('category', 'N/A')}\n")

    # Hybrid search
    logger.info(f"\nC. Hybrid Search for: '{query}'")
    logger.info("%s", "-" * 70)
    hybrid_results = hybrid_pipeline.run({"text_embedder": {"text": query}, "keyword_retriever": {"query": query}})
    logger.info(f"Found {len(hybrid_results['joiner']['documents'])} results:\n")
    for i, doc in enumerate(hybrid_results["joiner"]["documents"][:3], 1):
        score = doc.score if doc.score is not None else 0.0
        logger.info(f"{i}. Score: {score:.4f}")
        logger.info(f"   {doc.content[:80]}...")
        logger.info(f"   Category: {doc.meta.get('category', 'N/A')}\n")

    # Example 2: Technical query
    logger.info("\n6. Example 2: Technical query with hybrid search")
    logger.info("%s", "=" * 70)
    query = "neural networks deep learning"

    results = hybrid_pipeline.run({"text_embedder": {"text": query}, "keyword_retriever": {"query": query}})

    logger.info(f"Query: '{query}'")
    logger.info(f"Found {len(results['joiner']['documents'])} results:\n")
    for i, doc in enumerate(results["joiner"]["documents"][:5], 1):
        score = doc.score if doc.score is not None else 0.0
        logger.info(f"{i}. Score: {score:.4f}")
        logger.info(f"   Content: {doc.content}")
        logger.info(f"   Metadata: {doc.meta}")
        logger.info("")

    # Example 3: Specific technology query
    logger.info("\n7. Example 3: Specific technology query")
    logger.info("%s", "=" * 70)
    query = "JavaScript React frontend"

    results = hybrid_pipeline.run({"text_embedder": {"text": query}, "keyword_retriever": {"query": query}})

    logger.info(f"Query: '{query}'")
    logger.info(f"Found {len(results['joiner']['documents'])} results:\n")
    for i, doc in enumerate(results["joiner"]["documents"][:5], 1):
        score = doc.score if doc.score is not None else 0.0
        logger.info(f"{i}. Score: {score:.4f}")
        logger.info(f"   Content: {doc.content}")
        logger.info("")

    logger.info("%s", "=" * 70)
    logger.info("Example completed successfully!")
    logger.info("%s", "=" * 70)
    logger.info("\nKey Takeaways:")
    logger.info("- Keyword search: Exact term matching, good for specific terms")
    logger.info("- Embedding search: Semantic similarity, good for concepts")
    logger.info("- Hybrid search: Combines both approaches for best results")
    logger.info("- Reciprocal rank fusion merges results intelligently")
    logger.info("- Hybrid search often provides more comprehensive results")


if __name__ == "__main__":
    main()
