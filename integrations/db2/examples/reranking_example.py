"""
Example: Using SentenceTransformersSimilarityRanker with DB2 retrievers for improved retrieval quality.

This example demonstrates how to use reranking to improve the quality of retrieved documents
by reordering them based on semantic similarity to the query.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import Secret

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever, DB2HybridRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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


def setup_document_store() -> DB2DocumentStore:
    """Initialize DB2 document store with sample documents."""
    document_store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        hostname=os.getenv("DB2_HOST"),
        port=int(os.getenv("DB2_PORT", "50000")),
        table_name="reranking_demo",
        embedding_dimension=384,
        distance_metric="cosine",
        recreate_table=True,
    )

    # Sample documents about programming languages
    documents = [
        Document(
            content="Python is a high-level programming language known for its simplicity and readability. "
            "It's widely used in data science, machine learning, and web development.",
            meta={"category": "programming", "language": "python", "difficulty": "beginner"},
        ),
        Document(
            content="Java is an object-oriented programming language designed for portability across platforms. "
            "It's commonly used in enterprise applications and Android development.",
            meta={"category": "programming", "language": "java", "difficulty": "intermediate"},
        ),
        Document(
            content="JavaScript is the programming language of the web, enabling interactive web pages. "
            "It runs in browsers and on servers with Node.js.",
            meta={"category": "programming", "language": "javascript", "difficulty": "beginner"},
        ),
        Document(
            content="Rust is a systems programming language focused on safety and performance. "
            "It prevents memory errors and provides zero-cost abstractions.",
            meta={"category": "programming", "language": "rust", "difficulty": "advanced"},
        ),
        Document(
            content="Go is a statically typed language designed for simplicity and efficiency. "
            "It's popular for building scalable network services and cloud applications.",
            meta={"category": "programming", "language": "go", "difficulty": "intermediate"},
        ),
        Document(
            content="C++ is a powerful language used for system software, game engines, and performance-critical applications. "
            "It offers low-level memory manipulation and high performance.",
            meta={"category": "programming", "language": "cpp", "difficulty": "advanced"},
        ),
        Document(
            content="TypeScript is a superset of JavaScript that adds static typing. "
            "It helps catch errors early and improves code maintainability in large projects.",
            meta={"category": "programming", "language": "typescript", "difficulty": "intermediate"},
        ),
        Document(
            content="Swift is Apple's programming language for iOS, macOS, and other Apple platforms. "
            "It's designed to be safe, fast, and expressive.",
            meta={"category": "programming", "language": "swift", "difficulty": "intermediate"},
        ),
    ]

    # Generate embeddings
    logger.info("Generating embeddings...")
    embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    embedder.warm_up()
    docs_with_embeddings = embedder.run(documents)["documents"]

    # Write to document store
    document_store.write_documents(docs_with_embeddings)
    logger.info("Stored %d documents", len(docs_with_embeddings))

    return document_store


def example_embedding_retrieval_with_reranking(document_store: DB2DocumentStore) -> None:
    """Example: Embedding retrieval + reranking."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Embedding Retrieval with Reranking")
    logger.info("=" * 80)

    # Create pipeline
    pipeline = Pipeline()

    # Add components
    pipeline.add_component(
        "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    pipeline.add_component(
        "retriever",
        DB2EmbeddingRetriever(document_store=document_store, top_k=5),  # Retrieve more for reranking
    )
    pipeline.add_component(
        "ranker",
        SentenceTransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=3),  # Rerank to top 3
    )

    # Connect components
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "ranker.documents")

    # Run query
    query = "What language is best for beginners learning to code?"
    logger.info("Query: %s", query)

    results = pipeline.run({"text_embedder": {"text": query}, "ranker": {"query": query}})

    logger.info("Top 3 Results (after reranking):")
    for i, doc in enumerate(results["ranker"]["documents"], 1):
        logger.info(
            "%d. Score: %.4f | Language: %s | Difficulty: %s",
            i,
            doc.score,
            doc.meta.get("language", "N/A"),
            doc.meta.get("difficulty", "N/A"),
        )
        logger.info("   Content: %s...", doc.content[:150])


def example_hybrid_retrieval_with_reranking(document_store: DB2DocumentStore) -> None:
    """Example: Hybrid retrieval + reranking."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Hybrid Retrieval with Reranking")
    logger.info("=" * 80)

    # Create pipeline
    pipeline = Pipeline()

    # Add components
    pipeline.add_component(
        "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    pipeline.add_component(
        "hybrid_retriever",
        DB2HybridRetriever(document_store=document_store, top_k=6),  # Retrieve more for reranking
    )
    pipeline.add_component(
        "ranker",
        SentenceTransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=3),  # Rerank to top 3
    )

    # Connect components
    pipeline.connect("text_embedder.embedding", "hybrid_retriever.query_embedding")
    pipeline.connect("hybrid_retriever.documents", "ranker.documents")

    # Run query
    query = "performance critical systems programming"
    logger.info("Query: %s", query)

    results = pipeline.run(
        {
            "text_embedder": {"text": query},
            "hybrid_retriever": {"query": query},
            "ranker": {"query": query},
        }
    )

    logger.info("Top 3 Results (after reranking):")
    for i, doc in enumerate(results["ranker"]["documents"], 1):
        logger.info(
            "%d. Score: %.4f | Language: %s | Difficulty: %s",
            i,
            doc.score,
            doc.meta.get("language", "N/A"),
            doc.meta.get("difficulty", "N/A"),
        )
        logger.info("   Content: %s...", doc.content[:150])


def example_filtered_retrieval_with_reranking(document_store: DB2DocumentStore) -> None:
    """Example: Filtered retrieval + reranking."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Filtered Retrieval with Reranking")
    logger.info("=" * 80)

    # Create pipeline
    pipeline = Pipeline()

    # Add components
    pipeline.add_component(
        "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    pipeline.add_component(
        "retriever",
        DB2EmbeddingRetriever(
            document_store=document_store,
            top_k=5,
            # Filter for beginner-friendly languages only
            filters={"field": "meta.difficulty", "operator": "==", "value": "beginner"},
        ),
    )
    pipeline.add_component(
        "ranker",
        SentenceTransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=2),
    )

    # Connect components
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "ranker.documents")

    # Run query
    query = "easy to learn programming language for web development"
    logger.info("Query: %s", query)
    logger.info("Filter: difficulty == 'beginner'")

    results = pipeline.run({"text_embedder": {"text": query}, "ranker": {"query": query}})

    logger.info("Top 2 Results (filtered + reranked):")
    for i, doc in enumerate(results["ranker"]["documents"], 1):
        logger.info(
            "%d. Score: %.4f | Language: %s | Difficulty: %s",
            i,
            doc.score,
            doc.meta.get("language", "N/A"),
            doc.meta.get("difficulty", "N/A"),
        )
        logger.info("   Content: %s...", doc.content[:150])


def main() -> None:
    """Run all reranking examples."""
    logger.info("=" * 80)
    logger.info("DB2 + SentenceTransformersSimilarityRanker Examples")
    logger.info("=" * 80)

    # Setup
    document_store = setup_document_store()

    # Run examples
    example_embedding_retrieval_with_reranking(document_store)
    example_hybrid_retrieval_with_reranking(document_store)
    example_filtered_retrieval_with_reranking(document_store)

    logger.info("=" * 80)
    logger.info("All examples completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
