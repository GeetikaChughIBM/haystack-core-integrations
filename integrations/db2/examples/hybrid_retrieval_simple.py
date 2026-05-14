# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple Hybrid Retrieval Example for DB2

This example demonstrates how to combine vector similarity search with
keyword-based search for better retrieval results.

Hybrid search is particularly useful when you want to:
- Find semantically similar content (vector search)
- Match exact keywords or phrases (keyword search)
- Get the best of both approaches

Requirements:
- DB2 database with vector support
- sentence-transformers library
- haystack-ai framework

Install:
pip install sentence-transformers haystack-ai ibm-db
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever, DB2KeywordRetriever
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

# Sample documents about running shoes
DOCUMENTS = [
    Document(content="Nike Air Max 90 features visible Air cushioning in the heel for comfort during runs."),
    Document(content="Adidas Ultraboost uses Boost foam technology for responsive energy return."),
    Document(content="New Balance Fresh Foam provides plush cushioning for long distance running."),
    Document(content="ASICS Gel-Kayano offers excellent stability and support for overpronators."),
    Document(content="Brooks Ghost is a neutral running shoe with smooth transitions."),
]


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the hybrid retrieval example."""

    logger.info("%s", "=" * 60)
    logger.info("DB2 Hybrid Retrieval - Simple Example")
    logger.info("%s", "=" * 60)

    # Step 1: Initialize document store
    logger.info("\n1. Initializing DB2 document store...")
    document_store = DB2DocumentStore(
        database="TESTDB",
        username=Secret.from_env_var("DB2_USER", strict=False) or Secret.from_token("db2inst1"),
        password=Secret.from_env_var("DB2_PASSWORD", strict=False) or Secret.from_token("password"),
        table_name="hybrid_demo",
        embedding_dimension=384,  # all-MiniLM-L6-v2 dimension
        recreate_table=True,
    )
    logger.info("✓ Document store ready")

    # Step 2: Index documents with embeddings
    logger.info("\n2. Indexing documents...")
    indexing = Pipeline()
    indexing.add_component(
        "embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )
    indexing.add_component("writer", DocumentWriter(document_store=document_store))
    indexing.connect("embedder", "writer")

    indexing.run({"embedder": {"documents": DOCUMENTS}})
    logger.info(f"✓ Indexed {len(DOCUMENTS)} documents")

    # Step 3: Create hybrid search pipeline
    logger.info("\n3. Creating hybrid search pipeline...")
    hybrid_pipeline = Pipeline()

    # Add text embedder for query
    hybrid_pipeline.add_component(
        "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    )

    # Add embedding retriever (semantic search)
    hybrid_pipeline.add_component("embedding_retriever", DB2EmbeddingRetriever(document_store=document_store, top_k=3))

    # Add keyword retriever (lexical search)
    hybrid_pipeline.add_component("keyword_retriever", DB2KeywordRetriever(document_store=document_store, top_k=3))

    # Add document joiner to combine results
    hybrid_pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=5))

    # Connect components
    hybrid_pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    hybrid_pipeline.connect("embedding_retriever.documents", "joiner.documents")
    hybrid_pipeline.connect("keyword_retriever.documents", "joiner.documents")

    logger.info("✓ Pipeline ready")

    # Step 4: Run searches
    logger.info("%s", "\n" + "=" * 60)
    logger.info("SEARCH EXAMPLES")
    logger.info("%s", "=" * 60)

    # Example 1: Semantic search works well
    logger.info("\n--- Example 1: Semantic Query ---")
    query1 = "comfortable cushioning for running"
    logger.info(f"Query: '{query1}'")
    logger.info("\nThis query benefits from semantic understanding.")

    results1 = hybrid_pipeline.run({"text_embedder": {"text": query1}, "keyword_retriever": {"query": query1}})

    logger.info("\nTop Results:")
    for i, doc in enumerate(results1["joiner"]["documents"][:3], 1):
        logger.info(f"{i}. {doc.content[:70]}...")
        logger.info(f"   Score: {doc.score:.4f}")

    # Example 2: Keyword search helps with specific terms
    logger.info("\n--- Example 2: Specific Brand Query ---")
    query2 = "Nike Air Max"
    logger.info(f"Query: '{query2}'")
    logger.info("\nThis query benefits from exact keyword matching.")

    results2 = hybrid_pipeline.run({"text_embedder": {"text": query2}, "keyword_retriever": {"query": query2}})

    logger.info("\nTop Results:")
    for i, doc in enumerate(results2["joiner"]["documents"][:3], 1):
        logger.info(f"{i}. {doc.content[:70]}...")
        logger.info(f"   Score: {doc.score:.4f}")

    # Example 3: Hybrid search combines both
    logger.info("\n--- Example 3: Hybrid Query ---")
    query3 = "Boost technology cushioning"
    logger.info(f"Query: '{query3}'")
    logger.info("\nThis query benefits from both semantic and keyword matching.")

    results3 = hybrid_pipeline.run({"text_embedder": {"text": query3}, "keyword_retriever": {"query": query3}})

    logger.info("\nTop Results:")
    for i, doc in enumerate(results3["joiner"]["documents"][:3], 1):
        logger.info(f"{i}. {doc.content[:70]}...")
        logger.info(f"   Score: {doc.score:.4f}")

    # Summary
    logger.info("%s", "\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("%s", "=" * 60)
    logger.info(
        "%s",
        """
Hybrid Search Benefits:
✓ Semantic understanding (vector search)
✓ Exact keyword matching (keyword search)
✓ Better recall and precision
✓ Robust to different query types

How It Works:
1. Vector Search: Finds semantically similar documents
2. Keyword Search: Finds documents with matching terms
3. Fusion: Combines results using Reciprocal Rank Fusion
4. Final Ranking: Returns top documents from both approaches

Use Cases:
- E-commerce product search
- Document retrieval
- Question answering
- Content recommendation
    """,
    )

    logger.info("\n✓ Example complete!")


if __name__ == "__main__":
    main()
