# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Real-world integration tests for DB2 Haystack integration.

These tests demonstrate practical use cases and require a real DB2 connection.
Run with: pytest tests/test_real_world_scenarios.py -v -m integration
"""

import os

import pytest
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever, DB2HybridRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore


@pytest.mark.integration
class TestProductCatalogSearch:
    """Test hybrid search on a product catalog - common e-commerce scenario."""

    @pytest.fixture
    def product_store(self):
        """Create a document store with product catalog."""
        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("Local DB2 credentials not available")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="product_catalog",
            embedding_dimension=384,
            distance_metric="cosine",
            recreate_table=True,
        )

        # Add sample products
        products = [
            Document(
                id="laptop_mbp",
                content="Apple MacBook Pro 16-inch M2 chip 32GB RAM 1TB SSD for developers",
                embedding=[0.1 + i * 0.001 for i in range(384)],
                meta={"category": "laptops", "brand": "Apple", "price": 2499.99, "in_stock": True, "rating": 4.8},
            ),
            Document(
                id="laptop_dell",
                content="Dell XPS 15 Intel i9 16GB RAM 512GB SSD business productivity",
                embedding=[0.2 + i * 0.001 for i in range(384)],
                meta={"category": "laptops", "brand": "Dell", "price": 1799.99, "in_stock": True, "rating": 4.5},
            ),
            Document(
                id="phone_iphone",
                content="Apple iPhone 15 Pro A17 chip 256GB titanium flagship smartphone",
                embedding=[0.15 + i * 0.001 for i in range(384)],
                meta={"category": "smartphones", "brand": "Apple", "price": 999.99, "in_stock": False, "rating": 4.7},
            ),
            Document(
                id="phone_samsung",
                content="Samsung Galaxy S24 Ultra Snapdragon 512GB premium Android phone",
                embedding=[0.25 + i * 0.001 for i in range(384)],
                meta={"category": "smartphones", "brand": "Samsung", "price": 1199.99, "in_stock": True, "rating": 4.6},
            ),
            Document(
                id="tablet_ipad",
                content="Apple iPad Pro 12.9-inch M2 chip 256GB creative work tablet",
                embedding=[0.12 + i * 0.001 for i in range(384)],
                meta={"category": "tablets", "brand": "Apple", "price": 1099.99, "in_stock": True, "rating": 4.9},
            ),
        ]

        store.write_documents(products)
        yield store

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_search_apple_laptop(self, product_store):
        """Search for Apple laptop - should prioritize MacBook Pro."""
        retriever = DB2HybridRetriever(
            document_store=product_store,
            top_k=3,
            embedding_weight=0.7,
            keyword_weight=0.3,
        )

        results = retriever.run(
            query="Apple laptop for development",
            query_embedding=[0.1 + i * 0.001 for i in range(384)],
        )

        assert len(results["documents"]) > 0
        # MacBook Pro should be top result
        assert results["documents"][0].id == "laptop_mbp"
        assert results["documents"][0].meta["brand"] == "Apple"

    def test_filter_by_price_range(self, product_store):
        """Filter products by price range."""
        retriever = DB2EmbeddingRetriever(document_store=product_store, top_k=10)

        results = retriever.run(
            query_embedding=[0.2 + i * 0.001 for i in range(384)],
            filters={"price": {"$gte": 1000, "$lte": 2000}},
        )

        assert len(results["documents"]) > 0
        for doc in results["documents"]:
            assert 1000 <= doc.meta["price"] <= 2000

    def test_filter_in_stock_high_rating(self, product_store):
        """Find in-stock products with high ratings."""
        retriever = DB2EmbeddingRetriever(document_store=product_store, top_k=10)

        results = retriever.run(
            query_embedding=[0.15 + i * 0.001 for i in range(384)],
            filters={"in_stock": True, "rating": {"$gte": 4.5}},
        )

        assert len(results["documents"]) > 0
        for doc in results["documents"]:
            assert doc.meta["in_stock"] is True
            assert doc.meta["rating"] >= 4.5


@pytest.mark.integration
class TestDocumentationSearch:
    """Test semantic search on technical documentation."""

    @pytest.fixture
    def docs_store(self):
        """Create a document store with technical docs."""
        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("Local DB2 credentials not available")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="tech_docs",
            embedding_dimension=384,
            distance_metric="cosine",
            recreate_table=True,
        )

        # Add technical documentation
        docs = [
            Document(
                id="ml_intro",
                content="Machine learning tutorial for beginners covering supervised and unsupervised learning",
                embedding=[0.1 + i * 0.001 for i in range(384)],
                meta={"topic": "machine_learning", "level": "beginner", "views": 1000, "rating": 4.5},
            ),
            Document(
                id="dl_advanced",
                content="Advanced deep learning techniques neural networks transformers attention mechanisms",
                embedding=[0.2 + i * 0.001 for i in range(384)],
                meta={"topic": "deep_learning", "level": "advanced", "views": 5000, "rating": 4.8},
            ),
            Document(
                id="nlp_basics",
                content="Natural language processing basics text preprocessing tokenization embeddings",
                embedding=[0.15 + i * 0.001 for i in range(384)],
                meta={"topic": "nlp", "level": "intermediate", "views": 2000, "rating": 4.2},
            ),
            Document(
                id="cv_fundamentals",
                content="Computer vision fundamentals image processing convolutional neural networks",
                embedding=[0.25 + i * 0.001 for i in range(384)],
                meta={"topic": "computer_vision", "level": "beginner", "views": 3000, "rating": 4.6},
            ),
        ]

        store.write_documents(docs)
        yield store

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_search_beginner_tutorials(self, docs_store):
        """Find beginner-level tutorials."""
        retriever = DB2EmbeddingRetriever(document_store=docs_store, top_k=10)

        results = retriever.run(
            query_embedding=[0.1 + i * 0.001 for i in range(384)],
            filters={"level": "beginner"},
        )

        assert len(results["documents"]) == 2
        for doc in results["documents"]:
            assert doc.meta["level"] == "beginner"

    def test_search_popular_content(self, docs_store):
        """Find popular content with high views."""
        retriever = DB2EmbeddingRetriever(document_store=docs_store, top_k=10)

        results = retriever.run(
            query_embedding=[0.2 + i * 0.001 for i in range(384)],
            filters={"views": {"$gte": 2000}},
        )

        assert len(results["documents"]) > 0
        for doc in results["documents"]:
            assert doc.meta["views"] >= 2000

    def test_complex_filter_combination(self, docs_store):
        """Test complex AND/OR filter combinations."""
        retriever = DB2EmbeddingRetriever(document_store=docs_store, top_k=10)

        # Find: (beginner OR high views) AND high rating
        results = retriever.run(
            query_embedding=[0.15 + i * 0.001 for i in range(384)],
            filters={
                "$and": [
                    {"rating": {"$gte": 4.5}},
                    {
                        "$or": [
                            {"level": "beginner"},
                            {"views": {"$gte": 4000}},
                        ]
                    },
                ]
            },
        )

        assert len(results["documents"]) > 0
        for doc in results["documents"]:
            assert doc.meta["rating"] >= 4.5
            assert doc.meta["level"] == "beginner" or doc.meta["views"] >= 4000


@pytest.mark.integration
class TestDuplicateHandling:
    """Test duplicate document handling with different policies."""

    @pytest.fixture
    def dup_store(self):
        """Create a document store for duplicate testing."""
        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("Local DB2 credentials not available")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="dup_test",
            embedding_dimension=384,
            distance_metric="cosine",
            recreate_table=True,
        )

        yield store

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_skip_policy(self, dup_store):
        """Test SKIP policy - keeps original document."""
        doc_v1 = Document(
            id="doc1",
            content="Version 1",
            embedding=[0.1] * 384,
            meta={"version": 1},
        )

        dup_store.write_documents([doc_v1])

        doc_v2 = Document(
            id="doc1",
            content="Version 2",
            embedding=[0.2] * 384,
            meta={"version": 2},
        )

        dup_store.write_documents([doc_v2], policy=DuplicatePolicy.SKIP)

        # Should still have version 1
        docs = dup_store.filter_documents()
        result = [d for d in docs if d.id == "doc1"]
        assert len(result) == 1
        assert result[0].meta["version"] == 1
        assert result[0].content == "Version 1"

    def test_overwrite_policy(self, dup_store):
        """Test OVERWRITE policy - replaces with new document."""
        doc_v1 = Document(
            id="doc2",
            content="Version 1",
            embedding=[0.1] * 384,
            meta={"version": 1},
        )

        dup_store.write_documents([doc_v1])

        doc_v2 = Document(
            id="doc2",
            content="Version 2",
            embedding=[0.2] * 384,
            meta={"version": 2},
        )

        dup_store.write_documents([doc_v2], policy=DuplicatePolicy.OVERWRITE)

        # Should now have version 2
        docs = dup_store.filter_documents()
        result = [d for d in docs if d.id == "doc2"]
        assert len(result) == 1
        assert result[0].meta["version"] == 2
        assert result[0].content == "Version 2"

    def test_fail_policy(self, dup_store):
        """Test FAIL policy - raises error on duplicate."""
        doc_v1 = Document(
            id="doc3",
            content="Version 1",
            embedding=[0.1] * 384,
            meta={"version": 1},
        )

        dup_store.write_documents([doc_v1])

        doc_v2 = Document(
            id="doc3",
            content="Version 2",
            embedding=[0.2] * 384,
            meta={"version": 2},
        )

        with pytest.raises(ValueError, match="exists"):
            dup_store.write_documents([doc_v2], policy=DuplicatePolicy.NONE)


@pytest.mark.integration
class TestBatchOperations:
    """Test batch write and delete operations."""

    @pytest.fixture
    def batch_store(self):
        """Create a document store for batch testing."""
        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("Local DB2 credentials not available")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="batch_test",
            embedding_dimension=384,
            distance_metric="cosine",
            recreate_table=True,
        )

        yield store

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_batch_write_100_documents(self, batch_store):
        """Write 100 documents in a batch."""
        docs = [
            Document(
                id=f"doc_{i}",
                content=f"Document {i} content",
                embedding=[0.1 * i % 10 + j * 0.001 for j in range(384)],
                meta={"index": i, "category": f"cat_{i % 5}"},
            )
            for i in range(100)
        ]

        written = batch_store.write_documents(docs)
        assert written == 100

        # Verify count
        assert batch_store.count_documents() == 100

    def test_batch_delete_by_filter(self, batch_store):
        """Delete documents by filter."""
        # Add documents
        docs = [
            Document(
                id=f"del_{i}",
                content=f"Content {i}",
                embedding=[0.1 * i + j * 0.001 for j in range(384)],
                meta={"category": "A" if i < 50 else "B"},
            )
            for i in range(100)
        ]

        batch_store.write_documents(docs)

        # Delete category A by getting IDs first
        docs_to_delete = batch_store.filter_documents(filters={"category": "A"})
        ids_to_delete = [d.id for d in docs_to_delete]
        batch_store.delete_documents(ids_to_delete)

        # Should have 50 documents left (category B)
        assert batch_store.count_documents() == 50

        remaining = batch_store.filter_documents()
        for doc in remaining:
            assert doc.meta["category"] == "B"


@pytest.mark.integration
class TestEmptyAndEdgeCases:
    """Test empty results and edge cases."""

    @pytest.fixture
    def edge_store(self):
        """Create a document store for edge case testing."""
        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("Local DB2 credentials not available")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="edge_test",
            embedding_dimension=384,
            distance_metric="cosine",
            recreate_table=True,
        )

        yield store

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_empty_store_retrieval(self, edge_store):
        """Retrieve from empty store returns empty list."""
        retriever = DB2EmbeddingRetriever(document_store=edge_store, top_k=10)

        results = retriever.run(query_embedding=[0.1] * 384)

        assert len(results["documents"]) == 0

    def test_filter_no_matches(self, edge_store):
        """Filter with no matches returns empty list."""
        doc = Document(
            id="test1",
            content="Test",
            embedding=[0.1] * 384,
            meta={"category": "A"},
        )

        edge_store.write_documents([doc])

        # Filter for non-existent category
        results = edge_store.filter_documents(filters={"category": "B"})

        assert len(results) == 0

    def test_delete_nonexistent_documents(self, edge_store):
        """Delete non-existent documents doesn't raise error."""
        # Should not raise error
        edge_store.delete_documents(["nonexistent1", "nonexistent2"])

        assert edge_store.count_documents() == 0
