# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.db2 import DB2DocumentStore


@pytest.mark.integration
class TestDB2DocumentStoreInit:
    """Test document store initialization with different connection methods.

    These tests require a real DB2 connection and are marked as integration tests.
    """

    def test_init_local_connection(self):
        """Test initialization with local 3-parameter connection."""

        # Skip if credentials not available
        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("Local DB2 credentials not available")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="test_init_local",
            embedding_dimension=384,
            recreate_table=False,
        )

        assert store.table_name == "test_init_local"
        assert store.embedding_dimension == 384
        assert store.distance_metric == "cosine"

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_init_remote_connection_with_params(self):
        """Test initialization with remote connection parameters."""

        # Skip if remote credentials not available
        if not all([os.getenv("DB2_REMOTE_HOST"), os.getenv("DB2_REMOTE_USER"), os.getenv("DB2_REMOTE_PASSWORD")]):
            pytest.skip("Remote DB2 credentials not available")

        remote_user = os.getenv("DB2_REMOTE_USER")
        remote_password = os.getenv("DB2_REMOTE_PASSWORD")

        store = DB2DocumentStore(
            database=os.getenv("DB2_REMOTE_DATABASE", "BLUDB"),
            hostname=os.getenv("DB2_REMOTE_HOST"),
            port=int(os.getenv("DB2_REMOTE_PORT", "32310")),
            username=Secret.from_token(remote_user) if remote_user else Secret.from_token(""),
            password=Secret.from_token(remote_password) if remote_password else Secret.from_token(""),
            table_name="test_init_remote",
            embedding_dimension=768,
            distance_metric="euclidean",
            recreate_table=False,
        )

        assert store.embedding_dimension == 768
        assert store.distance_metric == "euclidean"

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_init_connection_string(self):
        """Test initialization with connection string."""

        # Skip if connection string not available
        conn_str = os.getenv("DB2_CONNECTION_STRING")
        if not conn_str:
            pytest.skip("DB2_CONNECTION_STRING not available")

        store = DB2DocumentStore(
            connection_string=Secret.from_token(conn_str),
            table_name="test_init_connstr",
            embedding_dimension=512,
            recreate_table=False,
        )

        assert store.embedding_dimension == 512

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_init_missing_credentials(self):
        """Test that initialization fails with missing credentials."""

        with pytest.raises(DocumentStoreError, match="Provide either"):
            DB2DocumentStore(embedding_dimension=384)

    def test_init_custom_table_name(self):
        """Test initialization with custom table name."""

        # Skip if credentials not available
        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("Local DB2 credentials not available")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="custom_table",
            embedding_dimension=384,
            recreate_table=False,
        )

        assert store.table_name == "custom_table"

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass


@pytest.mark.integration
class TestDB2DocumentStoreIntegration:
    """Integration tests requiring actual DB2 connection."""

    def test_write_documents(self, document_store):
        """Test writing documents to the store."""
        docs = [
            Document(
                id="doc1",
                content="First document",
                embedding=[0.1] * document_store.embedding_dimension,
                meta={"category": "test", "priority": 1},
            ),
            Document(
                id="doc2",
                content="Second document",
                embedding=[0.2] * document_store.embedding_dimension,
                meta={"category": "test", "priority": 2},
            ),
        ]

        written = document_store.write_documents(docs)
        assert written == 2

        count = document_store.count_documents()
        assert count == 2

    def test_write_documents_duplicate_fail(self, document_store):
        """Test that duplicate documents raise error with FAIL policy."""
        doc = Document(
            id="doc1",
            content="Test document",
            embedding=[0.1] * document_store.embedding_dimension,
        )

        document_store.write_documents([doc])

        with pytest.raises(ValueError, match="exists"):
            document_store.write_documents([doc], policy=DuplicatePolicy.NONE)

    def test_write_documents_duplicate_skip(self, document_store):
        """Test that duplicate documents are skipped with SKIP policy."""
        doc1 = Document(
            id="doc1",
            content="Original content",
            embedding=[0.1] * document_store.embedding_dimension,
        )
        doc2 = Document(
            id="doc1",
            content="Updated content",
            embedding=[0.2] * document_store.embedding_dimension,
        )

        written1 = document_store.write_documents([doc1])
        assert written1 == 1

        written2 = document_store.write_documents([doc2], policy=DuplicatePolicy.SKIP)
        assert written2 == 0

        # Verify original content is preserved
        docs = document_store.filter_documents()
        assert len(docs) == 1
        assert docs[0].content == "Original content"

    def test_write_documents_duplicate_overwrite(self, document_store):
        """Test that duplicate documents are overwritten with OVERWRITE policy."""
        doc1 = Document(
            id="doc1",
            content="Original content",
            embedding=[0.1] * document_store.embedding_dimension,
        )
        doc2 = Document(
            id="doc1",
            content="Updated content",
            embedding=[0.2] * document_store.embedding_dimension,
        )

        document_store.write_documents([doc1])
        document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE)

        # Verify content is updated
        docs = document_store.filter_documents()
        assert len(docs) == 1
        assert docs[0].content == "Updated content"

    def test_write_documents_missing_embedding(self, document_store):
        """Test that documents without embeddings raise error."""
        doc = Document(id="doc1", content="Test document")

        with pytest.raises(ValueError, match="missing embedding"):
            document_store.write_documents([doc])

    def test_write_documents_wrong_dimension(self, document_store):
        """Test that documents with wrong embedding dimension raise error."""
        doc = Document(
            id="doc1",
            content="Test document",
            embedding=[0.1] * 100,  # Wrong dimension
        )

        with pytest.raises(ValueError, match="dimension mismatch"):
            document_store.write_documents([doc])

    def test_count_documents(self, document_store):
        """Test counting documents."""
        assert document_store.count_documents() == 0

        docs = [
            Document(
                id=f"doc{i}",
                content=f"Document {i}",
                embedding=[0.1 * i] * document_store.embedding_dimension,
            )
            for i in range(5)
        ]

        document_store.write_documents(docs)
        assert document_store.count_documents() == 5

    def test_filter_documents_no_filter(self, document_store):
        """Test filtering documents without filters returns all."""
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Document {i}",
                embedding=[0.1 * i] * document_store.embedding_dimension,
                meta={"index": i},
            )
            for i in range(3)
        ]

        document_store.write_documents(docs)
        filtered = document_store.filter_documents()

        assert len(filtered) == 3

    def test_filter_documents_with_filter(self, document_store):
        """Test filtering documents with metadata filters."""
        docs = [
            Document(
                id="doc1",
                content="AI document",
                embedding=[0.1] * document_store.embedding_dimension,
                meta={"category": "AI", "priority": 1},
            ),
            Document(
                id="doc2",
                content="Programming document",
                embedding=[0.2] * document_store.embedding_dimension,
                meta={"category": "Programming", "priority": 2},
            ),
            Document(
                id="doc3",
                content="Another AI document",
                embedding=[0.3] * document_store.embedding_dimension,
                meta={"category": "AI", "priority": 3},
            ),
        ]

        document_store.write_documents(docs)

        # Filter by category
        filtered = document_store.filter_documents(filters={"category": "AI"})
        assert len(filtered) == 2
        assert all(doc.meta["category"] == "AI" for doc in filtered)

    def test_delete_documents(self, document_store):
        """Test deleting documents by ID."""
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Document {i}",
                embedding=[0.1 * i] * document_store.embedding_dimension,
            )
            for i in range(5)
        ]

        document_store.write_documents(docs)
        assert document_store.count_documents() == 5

        document_store.delete_documents(["doc1", "doc3"])
        assert document_store.count_documents() == 3

        remaining = document_store.filter_documents()
        remaining_ids = {doc.id for doc in remaining}
        assert remaining_ids == {"doc0", "doc2", "doc4"}

    def test_delete_documents_empty_list(self, document_store):
        """Test that deleting empty list does nothing."""
        docs = [
            Document(
                id="doc1",
                content="Test",
                embedding=[0.1] * document_store.embedding_dimension,
            )
        ]

        document_store.write_documents(docs)
        document_store.delete_documents([])
        assert document_store.count_documents() == 1

    def test_special_characters_in_content(self, document_store):
        """Test handling of special characters in content."""
        doc = Document(
            id="doc1",
            content="Content with 'single quotes' and \"double quotes\" and \\ backslash",
            embedding=[0.1] * document_store.embedding_dimension,
            meta={"key": "value with 'quotes'"},
        )

        document_store.write_documents([doc])
        retrieved = document_store.filter_documents()

        assert len(retrieved) == 1
        assert retrieved[0].content == doc.content

    def test_to_dict_and_from_dict(self, document_store):
        """Test serialization and deserialization."""
        # Serialize
        data = document_store.to_dict()

        assert "type" in data
        assert "init_parameters" in data
        assert data["init_parameters"]["table_name"] == document_store.table_name
        assert data["init_parameters"]["embedding_dimension"] == document_store.embedding_dimension

        # Deserialize
        restored = DB2DocumentStore.from_dict(data)

        assert restored.table_name == document_store.table_name
        assert restored.embedding_dimension == document_store.embedding_dimension
        assert restored.distance_metric == document_store.distance_metric

    def test_batch_size_parameter(self, document_store):
        """Test that batch_size parameter is properly set."""
        # Test default batch_size
        assert document_store.batch_size == 1000

        # Test custom batch_size

        custom_store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="test_batch_size",
            embedding_dimension=384,
            batch_size=500,
            recreate_table=False,
        )

        assert custom_store.batch_size == 500

        # Cleanup
        try:
            custom_store._drop_table_if_exists()
        except Exception:
            pass

    def test_delete_by_filter_standard_name(self, document_store):
        """Test delete_by_filter method (standard Haystack name)."""
        # Write test documents
        docs = [
            Document(
                id="del1",
                content="Document to delete",
                embedding=[0.1] * document_store.embedding_dimension,
                meta={"category": "temp", "priority": 1},
            ),
            Document(
                id="del2",
                content="Document to keep",
                embedding=[0.2] * document_store.embedding_dimension,
                meta={"category": "keep", "priority": 2},
            ),
        ]

        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        # Delete using standard name
        document_store.delete_by_filter(filters={"category": "temp"})

        # Verify deletion
        remaining = document_store.filter_documents()
        assert len(remaining) == 1
        assert remaining[0].id == "del2"

    def test_delete_documents_by_filters_alias(self, document_store):
        """Test delete_documents_by_filters method (alias for compatibility)."""
        # Write test documents
        docs = [
            Document(
                id="del3",
                content="Document to delete",
                embedding=[0.1] * document_store.embedding_dimension,
                meta={"category": "temp", "priority": 1},
            ),
            Document(
                id="del4",
                content="Document to keep",
                embedding=[0.2] * document_store.embedding_dimension,
                meta={"category": "keep", "priority": 2},
            ),
        ]

        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        # Delete using alias name
        document_store.delete_documents_by_filters(filters={"category": "temp"})

        # Verify deletion
        remaining = document_store.filter_documents()
        assert len(remaining) == 1
        assert remaining[0].id == "del4"


@pytest.mark.integration
class TestDB2DocumentStoreRemoteConnection:
    """Tests specifically for remote DB2 connections."""

    def test_remote_connection_with_credentials(self, document_store_remote):
        """Test remote connection with provided credentials."""
        # Write a test document
        doc = Document(
            id="remote_test",
            content="Testing remote connection",
            embedding=[0.1] * document_store_remote.embedding_dimension,
        )

        written = document_store_remote.write_documents([doc])
        assert written == 1

        count = document_store_remote.count_documents()
        assert count == 1

    def test_connection_string_connection(self, document_store_connection_string):
        """Test connection using connection string."""
        doc = Document(
            id="conn_str_test",
            content="Testing connection string",
            embedding=[0.1] * document_store_connection_string.embedding_dimension,
        )

        written = document_store_connection_string.write_documents([doc])
        assert written == 1

        count = document_store_connection_string.count_documents()
        assert count == 1
