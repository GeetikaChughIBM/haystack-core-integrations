# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Db2KeywordRetriever using live DB2 instance."""

import sys

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.ibm_db import Db2KeywordRetriever
from haystack_integrations.document_stores.ibm_db import Db2ConnectionConfig, Db2DocumentStore

# DB2 connection configuration for docker-compose DB2 instance
DB2_CONFIG = Db2ConnectionConfig(
    database="testdb",
    hostname="localhost",
    port=50000,
    username="db2inst1",
    password="Passw0rd123!",
    protocol="TCPIP",
)

# Use Python-version-specific table name to avoid conflicts
TEST_TABLE_NAME = f"test_retriever_kw_{sys.version_info.major}_{sys.version_info.minor}"


@pytest.fixture
def document_store():
    """Create a fresh document store for each test."""
    store = Db2DocumentStore(
        connection_config=DB2_CONFIG,
        table_name=TEST_TABLE_NAME,
        embedding_dim=4,  # Small dimension for testing
        distance_metric="COSINE",
        recreate_table=True,
    )
    yield store
    # Cleanup after test
    try:
        conn = store._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {store.table_name}")
            conn.commit()
    except Exception:
        pass


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content="Python programming language is great for data science",
            meta={"category": "programming", "language": "python"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        ),
        Document(
            id="doc2",
            content="Java development for enterprise applications",
            meta={"category": "programming", "language": "java"},
            embedding=[0.5, 0.6, 0.7, 0.8],
        ),
        Document(
            id="doc3",
            content="Data science with Python and machine learning",
            meta={"category": "data-science", "language": "python"},
            embedding=[0.15, 0.25, 0.35, 0.45],
        ),
    ]


@pytest.mark.integration
class TestDb2KeywordRetriever:
    """Test Db2KeywordRetriever with actual DB2 database."""

    def test_retriever_initialization(self, document_store):
        """Test retriever initialization."""
        retriever = Db2KeywordRetriever(document_store=document_store, top_k=5)
        assert retriever.document_store == document_store
        assert retriever.top_k == 5
        assert retriever.filters == {}
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_invalid_document_store_raises_type_error(self):
        """Test that invalid document store raises TypeError."""
        with pytest.raises(TypeError, match="must be an instance of Db2DocumentStore"):
            Db2KeywordRetriever(document_store="not_a_store")

    def test_run_basic_retrieval(self, document_store, sample_documents):
        """Test basic keyword retrieval using fallback LIKE search."""
        document_store.write_documents(sample_documents)
        retriever = Db2KeywordRetriever(document_store=document_store, top_k=5)

        # Search for "Python" - should match doc1 and doc3
        result = retriever.run(query="Python")

        assert "documents" in result
        docs = result["documents"]
        assert len(docs) >= 2  # At least doc1 and doc3
        assert all(isinstance(doc, Document) for doc in docs)
        # Check that Python is in the content
        assert all("Python" in doc.content or "python" in doc.content.lower() for doc in docs)

    def test_run_with_filters(self, document_store, sample_documents):
        """Test retrieval with metadata filters."""
        document_store.write_documents(sample_documents)
        retriever = Db2KeywordRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            top_k=5,
        )

        result = retriever.run(query="programming")
        docs = result["documents"]

        # Should only return Python documents
        assert len(docs) >= 1
        assert all(doc.meta.get("language") == "python" for doc in docs)

    def test_run_with_runtime_filters_replace_policy(self, document_store, sample_documents):
        """Test that runtime filters replace constructor filters with REPLACE policy."""
        document_store.write_documents(sample_documents)
        retriever = Db2KeywordRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            filter_policy=FilterPolicy.REPLACE,
            top_k=5,
        )

        # Runtime filter should replace constructor filter
        runtime_filters = {"operator": "==", "field": "meta.language", "value": "java"}
        result = retriever.run(query="development", filters=runtime_filters)
        docs = result["documents"]

        # Should only return Java documents (runtime filter)
        if len(docs) > 0:
            assert all(doc.meta.get("language") == "java" for doc in docs)

    def test_run_with_runtime_filters_merge_policy(self, document_store, sample_documents):
        """Test that runtime filters merge with constructor filters with MERGE policy."""
        document_store.write_documents(sample_documents)
        retriever = Db2KeywordRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.category", "value": "programming"},
            filter_policy=FilterPolicy.MERGE,
            top_k=5,
        )

        # Runtime filter should merge with constructor filter
        runtime_filters = {"operator": "==", "field": "meta.language", "value": "python"}
        result = retriever.run(query="programming", filters=runtime_filters)
        docs = result["documents"]

        # Should return only Python programming documents (both filters applied)
        if len(docs) > 0:
            assert all(doc.meta.get("category") == "programming" for doc in docs)
            assert all(doc.meta.get("language") == "python" for doc in docs)

    def test_run_top_k_override(self, document_store, sample_documents):
        """Test that runtime top_k overrides constructor top_k."""
        document_store.write_documents(sample_documents)
        retriever = Db2KeywordRetriever(document_store=document_store, top_k=10)

        result = retriever.run(query="data", top_k=1)
        docs = result["documents"]

        assert len(docs) <= 1

    def test_run_empty_store(self, document_store):
        """Test retrieval from empty store."""
        retriever = Db2KeywordRetriever(document_store=document_store, top_k=5)
        result = retriever.run(query="test")

        assert result["documents"] == []

    def test_run_no_matches(self, document_store, sample_documents):
        """Test retrieval with query that matches no documents."""
        document_store.write_documents(sample_documents)
        retriever = Db2KeywordRetriever(document_store=document_store, top_k=5)

        result = retriever.run(query="nonexistentquery12345")

        assert result["documents"] == []

    def test_to_dict(self, document_store):
        """Test serialization to dictionary."""
        retriever = Db2KeywordRetriever(
            document_store=document_store,
            top_k=7,
            filters={"operator": "==", "field": "meta.x", "value": "y"},
        )
        d = retriever.to_dict()

        assert d["init_parameters"]["top_k"] == 7
        assert d["init_parameters"]["filters"] == {"operator": "==", "field": "meta.x", "value": "y"}
        assert d["init_parameters"]["filter_policy"] == "replace"
        assert "document_store" in d["init_parameters"]

    def test_from_dict(self, document_store):
        """Test deserialization from dictionary."""
        retriever = Db2KeywordRetriever(
            document_store=document_store,
            top_k=7,
            filters={"operator": "==", "field": "meta.x", "value": "y"},
        )
        d = retriever.to_dict()

        restored = Db2KeywordRetriever.from_dict(d)
        assert restored.top_k == 7
        assert restored.filters == {"operator": "==", "field": "meta.x", "value": "y"}
        assert restored.filter_policy == FilterPolicy.REPLACE
        assert restored.document_store.table_name == document_store.table_name
        assert restored.document_store.embedding_dim == 4

    @pytest.mark.asyncio
    async def test_run_async(self, document_store, sample_documents):
        """Test async retrieval."""
        document_store.write_documents(sample_documents)
        retriever = Db2KeywordRetriever(document_store=document_store, top_k=5)

        result = await retriever.run_async(query="Python")

        assert "documents" in result
        docs = result["documents"]
        assert len(docs) >= 1
        assert all(isinstance(doc, Document) for doc in docs)


# Made with Bob
