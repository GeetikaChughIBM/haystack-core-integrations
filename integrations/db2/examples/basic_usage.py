"""
Basic usage example for DB2 Haystack integration.

This example demonstrates:
- Connecting to DB2 with Secret objects
- Writing documents with embeddings
- Filtering documents by metadata
- Counting and deleting documents

Prerequisites:
- DB2 database running (v12.1.2+ with vector support)
- Environment variables set (see .env.example)
- Install: pip install db2-haystack
"""

import logging
import os

from dotenv import load_dotenv
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.document_stores.db2 import DB2DocumentStore

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run basic DB2 document store operations."""

    logger.info("%s", "=" * 70)
    logger.info("DB2 Haystack Integration - Basic Usage Example")
    logger.info("%s", "=" * 70)

    # Initialize document store with Secret objects for secure credential management
    logger.info("\n1. Initializing DB2 Document Store...")
    document_store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        hostname=os.getenv("DB2_HOST", "localhost"),
        port=int(os.getenv("DB2_PORT", "50000")),
        table_name="example_documents",
        embedding_dimension=384,
        distance_metric="cosine",
        recreate_table=True,  # Start fresh for this example
    )
    logger.info("✓ Document store initialized")

    # Create sample documents with embeddings
    logger.info("\n2. Creating sample documents...")
    documents = [
        Document(
            id="doc_1",
            content="Paris is the capital and largest city of France.",
            embedding=[0.1] * 384,  # Simplified embedding for example
            meta={"category": "geography", "country": "France", "year": 2024},
        ),
        Document(
            id="doc_2",
            content="Berlin is the capital and largest city of Germany.",
            embedding=[0.2] * 384,
            meta={"category": "geography", "country": "Germany", "year": 2024},
        ),
        Document(
            id="doc_3",
            content="Python is a high-level programming language.",
            embedding=[0.3] * 384,
            meta={"category": "technology", "topic": "programming", "year": 2024},
        ),
        Document(
            id="doc_4",
            content="Machine learning is a subset of artificial intelligence.",
            embedding=[0.4] * 384,
            meta={"category": "technology", "topic": "AI", "year": 2024},
        ),
        Document(
            id="doc_5",
            content="Rome is the capital city of Italy.",
            embedding=[0.5] * 384,
            meta={"category": "geography", "country": "Italy", "year": 2024},
        ),
    ]
    logger.info(f"✓ Created {len(documents)} documents")

    # Write documents to DB2
    logger.info("\n3. Writing documents to DB2...")
    written_count = document_store.write_documents(documents)
    logger.info(f"✓ Successfully wrote {written_count} documents")

    # Count all documents
    logger.info("\n4. Counting documents...")
    total_count = document_store.count_documents()
    logger.info(f"✓ Total documents in store: {total_count}")

    # Filter documents by category
    logger.info("\n5. Filtering documents by category='geography'...")
    geo_docs = document_store.filter_documents(filters={"field": "category", "operator": "==", "value": "geography"})
    logger.info(f"✓ Found {len(geo_docs)} geography documents:")
    for doc in geo_docs:
        logger.info(f"   - {doc.id}: {doc.content[:50]}...")

    # Filter with complex conditions (AND operator)
    logger.info("\n6. Filtering with complex conditions (category='technology' AND topic='AI')...")
    tech_docs = document_store.filter_documents(
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "category", "operator": "==", "value": "technology"},
                {"field": "topic", "operator": "==", "value": "AI"},
            ],
        }
    )
    logger.info(f"✓ Found {len(tech_docs)} matching documents:")
    for doc in tech_docs:
        logger.info(f"   - {doc.id}: {doc.content[:50]}...")

    # Filter with OR operator
    logger.info("\n7. Filtering with OR operator (country='France' OR country='Italy')...")
    country_docs = document_store.filter_documents(
        filters={
            "operator": "OR",
            "conditions": [
                {"field": "country", "operator": "==", "value": "France"},
                {"field": "country", "operator": "==", "value": "Italy"},
            ],
        }
    )
    logger.info(f"✓ Found {len(country_docs)} matching documents:")
    for doc in country_docs:
        logger.info(f"   - {doc.id}: {doc.content[:50]}...")

    # Delete specific documents
    logger.info("\n8. Deleting documents...")
    document_store.delete_documents(["doc_1", "doc_2"])
    remaining_count = document_store.count_documents()
    logger.info(f"✓ Deleted 2 documents. Remaining: {remaining_count}")

    # Clean up - delete all documents
    logger.info("\n9. Cleaning up...")
    all_doc_ids = [doc.id for doc in document_store.filter_documents()]
    document_store.delete_documents(all_doc_ids)
    final_count = document_store.count_documents()
    logger.info(f"✓ Cleanup complete. Final count: {final_count}")

    logger.info("%s", "\n" + "=" * 70)
    logger.info("Example completed successfully!")
    logger.info("%s", "=" * 70)


if __name__ == "__main__":
    main()
