# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
import time

import pytest
from haystack import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.db2 import DB2DocumentStore


@pytest.mark.integration
class TestDB2DocumentStoreInit:
    """Test document store initialization with supported connection methods.

    These tests require a real DB2 connection and are marked as integration tests.
    """

    def test_init_env_connection(self):
        """Test initialization with environment-backed connection parameters."""

        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("DB2 credentials not available")

        use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            hostname=os.getenv("DB2_HOSTNAME"),
            port=int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000")),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="test_init_env",
            embedding_dimension=384,
            recreate_table=False,
            use_ssl=use_ssl,
            ssl_certificate=os.getenv("DB2_SSL_CERTIFICATE") or os.getenv("DB2_SSL_CERT_PATH"),
        )

        assert store.table_name == "test_init_env"
        assert store.embedding_dimension == 384
        assert store.distance_metric == "cosine"

        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_init_connection_with_explicit_hostname_and_port(self):
        """Test initialization with explicit connection parameters."""

        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("DB2 credentials not available")

        use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            hostname=os.getenv("DB2_HOSTNAME"),
            port=int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000")),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="test_init_params",
            embedding_dimension=768,
            distance_metric="euclidean",
            recreate_table=False,
            use_ssl=use_ssl,
            ssl_certificate=os.getenv("DB2_SSL_CERTIFICATE") or os.getenv("DB2_SSL_CERT_PATH"),
        )

        assert store.embedding_dimension == 768
        assert store.distance_metric == "euclidean"

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
            pytest.skip("DB2 credentials not available")

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
class TestDB2DocumentStoreConnectionModes:
    """Tests for the supported DB2 connection flows."""

    def test_env_connection_with_credentials(self, document_store_env):
        """Test environment-backed connection with provided credentials."""
        doc = Document(
            id="env_test",
            content="Testing env connection",
            embedding=[0.1] * document_store_env.embedding_dimension,
        )

        written = document_store_env.write_documents([doc])
        assert written == 1

        count = document_store_env.count_documents()
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


@pytest.mark.integration
class TestDB2DocumentStoreSSL:
    """Tests for SSL/TLS connection support."""

    def test_ssl_connection_parameters(self):
        """Test that SSL parameters are properly set in connection string."""
        # Skip if SSL credentials not available
        if not os.getenv("DB2_SSL_ENABLED"):
            pytest.skip("SSL-enabled DB2 not available for testing")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            hostname=os.getenv("DB2_HOSTNAME"),
            port=int(os.getenv("DB2_SSL_PORT", "50001")),
            table_name="test_ssl",
            embedding_dimension=384,
            use_ssl=True,
            recreate_table=False,
        )

        # Verify SSL is in connection string
        assert store._connection_string is not None
        assert "SECURITY=SSL" in store._connection_string.upper()

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_ssl_with_certificate(self):
        """Test SSL connection with certificate path."""
        # Skip if SSL credentials not available
        if not os.getenv("DB2_SSL_ENABLED"):
            pytest.skip("SSL-enabled DB2 not available for testing")

        cert_path = os.getenv("DB2_SSL_CERT_PATH", "/path/to/cert.pem")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            hostname=os.getenv("DB2_HOSTNAME"),
            port=int(os.getenv("DB2_SSL_PORT", "50001")),
            table_name="test_ssl_cert",
            embedding_dimension=384,
            use_ssl=True,
            ssl_certificate=cert_path,
            recreate_table=False,
        )

        # Verify SSL and certificate are in connection string
        assert store._connection_string is not None
        assert "SECURITY=SSL" in store._connection_string.upper()
        assert f"SSLSERVERCERTIFICATE={cert_path}" in store._connection_string

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_ssl_connection_string_direct(self):
        """Test SSL via direct connection string."""
        # Skip if SSL connection string not available
        ssl_conn_str = os.getenv("DB2_SSL_CONNECTION_STRING")
        if not ssl_conn_str:
            pytest.skip("DB2_SSL_CONNECTION_STRING not available")

        store = DB2DocumentStore(
            connection_string=Secret.from_token(ssl_conn_str),
            table_name="test_ssl_direct",
            embedding_dimension=384,
            recreate_table=False,
        )

        # Verify connection string contains SSL
        assert store._connection_string is not None
        assert "SECURITY=SSL" in store._connection_string.upper()

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass


@pytest.mark.integration
class TestDB2DocumentStoreAsync:
    """Tests for async method behavior."""

    def test_count_documents_async_raises_not_implemented(self, document_store_local):
        """Test that count_documents_async raises NotImplementedError."""
        import asyncio

        async def test_async():
            with pytest.raises(NotImplementedError) as exc_info:
                await document_store_local.count_documents_async()
            assert "Async operations are not yet supported" in str(exc_info.value)

        asyncio.run(test_async())

    def test_write_documents_async_raises_not_implemented(self, document_store_local):
        """Test that write_documents_async raises NotImplementedError."""
        import asyncio
        from haystack.document_stores.types import DuplicatePolicy

        async def test_async():
            doc = Document(
                id="async_test",
                content="Test async",
                embedding=[0.1] * document_store_local.embedding_dimension,
            )
            with pytest.raises(NotImplementedError) as exc_info:
                await document_store_local.write_documents_async([doc], policy=DuplicatePolicy.NONE)
            assert "Async operations are not yet supported" in str(exc_info.value)

        asyncio.run(test_async())

    def test_delete_documents_async_raises_not_implemented(self, document_store_local):
        """Test that delete_documents_async raises NotImplementedError."""
        import asyncio

        async def test_async():
            with pytest.raises(NotImplementedError) as exc_info:
                await document_store_local.delete_documents_async(["test_id"])
            assert "Async operations are not yet supported" in str(exc_info.value)

        asyncio.run(test_async())

    def test_filter_documents_async_raises_not_implemented(self, document_store_local):
        """Test that filter_documents_async raises NotImplementedError."""
        import asyncio

        async def test_async():
            with pytest.raises(NotImplementedError) as exc_info:
                await document_store_local.filter_documents_async()
            assert "Async operations are not yet supported" in str(exc_info.value)

        asyncio.run(test_async())

    def test_async_methods_have_clear_error_messages(self, document_store_local):
        """Test that all async methods provide clear error messages."""
        import asyncio

        async def test_all_async():
            methods = [
                document_store_local.count_documents_async(),
                document_store_local.write_documents_async([]),
                document_store_local.delete_documents_async([]),
                document_store_local.filter_documents_async(),
            ]

            for method in methods:
                with pytest.raises(NotImplementedError) as exc_info:
                    await method
                error_msg = str(exc_info.value)
                assert "Async operations are not yet supported" in error_msg
                assert "Use synchronous methods instead" in error_msg

        asyncio.run(test_all_async())


@pytest.mark.integration
class TestDB2DocumentStoreSSLConnection:
    """Tests for SSL/TLS connection parameters."""

    def test_ssl_connection_with_certificate(self):
        """Test SSL connection with certificate path."""
        if not all([os.getenv("DB2_HOSTNAME"), os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD")]):
            pytest.skip("DB2 credentials not available for SSL testing")

        ssl_cert_path = os.getenv("DB2_SSL_CERTIFICATE")
        if not ssl_cert_path:
            pytest.skip("DB2_SSL_CERTIFICATE not set")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            hostname=os.getenv("DB2_HOSTNAME"),
            port=int(os.getenv("DB2_SSL_PORT", "50001")),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="test_ssl_cert",
            embedding_dimension=384,
            use_ssl=True,
            ssl_certificate=ssl_cert_path,
            recreate_table=False,
        )

        count = store.count_documents()
        assert count >= 0

        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_ssl_connection_with_security_param(self):
        """Test SSL connection with SSL enabled."""
        if not all([os.getenv("DB2_HOSTNAME"), os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD")]):
            pytest.skip("DB2 credentials not available for SSL testing")

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            hostname=os.getenv("DB2_HOSTNAME"),
            port=int(os.getenv("DB2_SSL_PORT", "50001")),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="test_ssl_security",
            embedding_dimension=384,
            use_ssl=True,
            recreate_table=False,
        )

        # Verify connection works
        count = store.count_documents()
        assert count >= 0

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass


@pytest.mark.integration
class TestDB2DocumentStoreAsyncMethods:
    """Tests for async methods (which delegate to sync methods in ibm_db)."""

    def test_count_documents_async(self, document_store):
        """Test async count_documents method."""
        # Write some documents
        docs = [
            Document(
                id=f"async_doc{i}",
                content=f"Async test document {i}",
                embedding=[0.1 * i] * document_store.embedding_dimension,
            )
            for i in range(3)
        ]
        document_store.write_documents(docs)

        # Test async count (should delegate to sync method)
        count = document_store.count_documents_async()
        assert count == 3

    def test_write_documents_async(self, document_store):
        """Test async write_documents method."""
        docs = [
            Document(
                id="async_write_1",
                content="Async write test",
                embedding=[0.1] * document_store.embedding_dimension,
            )
        ]

        # Test async write (should delegate to sync method)
        written = document_store.write_documents_async(docs)
        assert written == 1

        # Verify document was written
        count = document_store.count_documents()
        assert count == 1

    def test_delete_documents_async(self, document_store):
        """Test async delete_documents method."""
        # Write a document
        doc = Document(
            id="async_delete_1",
            content="To be deleted",
            embedding=[0.1] * document_store.embedding_dimension,
        )
        document_store.write_documents([doc])

        # Test async delete (should delegate to sync method)
        document_store.delete_documents_async(["async_delete_1"])

        # Verify deletion
        count = document_store.count_documents()
        assert count == 0

    def test_filter_documents_async(self, document_store):
        """Test async filter_documents method."""
        docs = [
            Document(
                id="async_filter_1",
                content="Test document",
                embedding=[0.1] * document_store.embedding_dimension,
                meta={"category": "test"},
            )
        ]
        document_store.write_documents(docs)

        # Test async filter (should delegate to sync method)
        filtered = document_store.filter_documents_async(filters={"category": "test"})

        assert len(filtered) == 1
        assert filtered[0].id == "async_filter_1"


@pytest.mark.integration
class TestDB2DocumentStoreSchemaSupport:
    """Tests for schema parameter support."""

    def test_init_with_schema(self):
        """Test initialization with custom schema."""
        # Skip if credentials not available
        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("Local DB2 credentials not available")

        # Use default schema (user's schema)
        schema = os.getenv("DB2_USER", "").upper()

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="test_schema_support",
            embedding_dimension=384,
            schema_name=schema,
            recreate_table=False,
        )

        assert store.schema_name == schema

        # Verify table operations work with schema
        count = store.count_documents()
        assert count >= 0

        # Cleanup
        try:
            store._drop_table_if_exists()
        except Exception:
            pass

    def test_schema_in_serialization(self):
        """Test that schema is preserved in serialization."""
        # Skip if credentials not available
        if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE")]):
            pytest.skip("Local DB2 credentials not available")

        schema = os.getenv("DB2_USER", "").upper()

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="test_schema_serial",
            embedding_dimension=384,
            schema_name=schema,
            recreate_table=False,
        )

        # Serialize
        data = store.to_dict()
        assert data["init_parameters"]["schema_name"] == schema

        # Deserialize
        restored = DB2DocumentStore.from_dict(data)
        assert restored.schema_name == schema

        # Cleanup
        try:
            store._drop_table_if_exists()
            restored._drop_table_if_exists()
        except Exception:
            pass


@pytest.mark.integration
class TestDB2DocumentStoreMetadataOperations:
    """Tests for metadata table operations."""

    def test_metadata_table_creation(self, document_store):
        """Test that metadata table is created."""
        # Metadata table should exist after document store initialization
        assert document_store._metadata_table_exists()

    def test_get_set_metadata(self, document_store):
        """Test getting and setting metadata."""
        # Set metadata
        document_store._set_metadata("test_key", "test_value")

        # Get metadata
        value = document_store._get_metadata("test_key")
        assert value == "test_value"

    def test_validate_model_consistency(self, document_store):
        """Test model consistency validation."""
        # First initialization should set metadata
        embedding_dim = document_store.embedding_dimension
        distance_metric = document_store.distance_metric

        # Validation should pass with same parameters
        document_store._validate_model_consistency()

        # Create new store with same table but different parameters
        # This should raise an error
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            conflicting_store = DB2DocumentStore(
                database=document_store._database,
                username=document_store._username,
                password=document_store._password,
                table_name=document_store.table_name,
                embedding_dimension=embedding_dim + 100,  # Different dimension
                distance_metric=distance_metric,
                recreate_table=False,
            )

    def test_metadata_fields_info(self, document_store):
        """Test getting metadata fields information."""
        # Write documents with various metadata types
        docs = [
            Document(
                id="meta_info_1",
                content="Test 1",
                embedding=[0.1] * document_store.embedding_dimension,
                meta={"category": "AI", "priority": 1, "active": True},
            ),
            Document(
                id="meta_info_2",
                content="Test 2",
                embedding=[0.2] * document_store.embedding_dimension,
                meta={"category": "ML", "priority": 2, "active": False},
            ),
        ]
        document_store.write_documents(docs)

        # Get metadata fields info
        fields_info = document_store.get_metadata_fields_info()

        # Should return information about metadata fields
        assert isinstance(fields_info, dict)
        assert "category" in fields_info
        assert "priority" in fields_info
        assert "active" in fields_info

    def test_metadata_field_unique_values(self, document_store):
        """Test getting unique values for a metadata field."""
        docs = [
            Document(
                id="unique_1",
                content="Test 1",
                embedding=[0.1] * document_store.embedding_dimension,
                meta={"category": "AI"},
            ),
            Document(
                id="unique_2",
                content="Test 2",
                embedding=[0.2] * document_store.embedding_dimension,
                meta={"category": "ML"},
            ),
            Document(
                id="unique_3",
                content="Test 3",
                embedding=[0.3] * document_store.embedding_dimension,
                meta={"category": "AI"},
            ),
        ]
        document_store.write_documents(docs)

        # Get unique values for category field
        unique_values = document_store.get_metadata_field_unique_values("category")

        assert isinstance(unique_values, list)
        assert set(unique_values) == {"AI", "ML"}

    def test_metadata_field_min_max(self, document_store):
        """Test getting min/max values for numeric metadata field."""
        docs = [
            Document(
                id="minmax_1",
                content="Test 1",
                embedding=[0.1] * document_store.embedding_dimension,
                meta={"priority": 1},
            ),
            Document(
                id="minmax_2",
                content="Test 2",
                embedding=[0.2] * document_store.embedding_dimension,
                meta={"priority": 5},
            ),
            Document(
                id="minmax_3",
                content="Test 3",
                embedding=[0.3] * document_store.embedding_dimension,
                meta={"priority": 3},
            ),
        ]
        document_store.write_documents(docs)

        # Get min/max for priority field
        min_val, max_val = document_store.get_metadata_field_min_max("priority")

        assert min_val == 1
        assert max_val == 5


@pytest.mark.integration
class TestDB2BulkInsertOptimization:
    """Test bulk insert optimization with both batch and individual modes."""

    def test_batch_insert_mode_default(self, document_store):
        """Test that batch insert is enabled by default."""
        assert document_store.use_batch_insert is True

    def test_batch_insert_mode_disabled(self):
        """Test creating store with batch insert disabled."""
        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="test_batch_disabled",
            embedding_dimension=384,
            use_batch_insert=False,
            recreate_table=True,
        )
        
        try:
            assert store.use_batch_insert is False
            
            # Write documents and verify they work
            docs = [
                Document(
                    id=f"test_{i}",
                    content=f"Test {i}",
                    embedding=[0.1 * i] * 384,
                )
                for i in range(10)
            ]
            
            written = store.write_documents(docs)
            assert written == 10
            assert store.count_documents() == 10
            
        finally:
            try:
                store._drop_table_if_exists()
            except Exception:
                pass

    def test_batch_insert_writes_correctly(self, document_store):
        """Test that batch insert writes documents correctly."""
        docs = [
            Document(
                id=f"batch_{i}",
                content=f"Document {i}",
                embedding=[0.1 * i] * document_store.embedding_dimension,
                meta={"index": i},
            )
            for i in range(50)
        ]
        
        written = document_store.write_documents(docs)
        assert written == 50
        assert document_store.count_documents() == 50
        
        # Verify documents are retrievable
        retrieved = document_store.filter_documents()
        assert len(retrieved) == 50

    def test_batch_insert_with_duplicates_skip(self, document_store):
        """Test batch insert with SKIP policy on duplicates."""
        # Write initial documents
        initial_docs = [
            Document(id=f"dup_{i}", content=f"Doc {i}", embedding=[0.1 * i] * document_store.embedding_dimension)
            for i in range(5)
        ]
        document_store.write_documents(initial_docs)
        
        # Try to write duplicates with SKIP
        duplicate_docs = [
            Document(id="dup_0", content="Duplicate", embedding=[0.1] * document_store.embedding_dimension),
            Document(id="dup_1", content="Duplicate", embedding=[0.2] * document_store.embedding_dimension),
        ]
        
        written = document_store.write_documents(duplicate_docs, policy=DuplicatePolicy.SKIP)
        assert written == 0
        assert document_store.count_documents() == 5

    def test_batch_insert_with_duplicates_overwrite(self, document_store):
        """Test batch insert with OVERWRITE policy."""
        # Write initial document
        initial = Document(id="overwrite_test", content="Original", embedding=[0.1] * document_store.embedding_dimension)
        document_store.write_documents([initial])
        
        # Overwrite with new content
        updated = Document(id="overwrite_test", content="Updated", embedding=[0.2] * document_store.embedding_dimension)
        written = document_store.write_documents([updated], policy=DuplicatePolicy.OVERWRITE)
        
        assert written == 1
        assert document_store.count_documents() == 1
        
        # Verify content was updated
        docs = document_store.filter_documents()
        assert docs[0].content == "Updated"

    def test_batch_insert_large_batch(self, document_store):
        """Test batch insert with documents exceeding batch_size."""
        # Create 250 documents (will be processed in multiple batches)
        docs = [
            Document(
                id=f"large_{i}",
                content=f"Document {i}",
                embedding=[0.001 * i] * document_store.embedding_dimension,
                meta={"batch": i // 100},
            )
            for i in range(250)
        ]
        
        written = document_store.write_documents(docs)
        assert written == 250
        assert document_store.count_documents() == 250

    def test_batch_insert_serialization(self):
        """Test that use_batch_insert is preserved in serialization."""
        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="test_serial",
            embedding_dimension=384,
            use_batch_insert=False,
            recreate_table=True,
        )
        
        try:
            # Serialize
            config = store.to_dict()
            assert "use_batch_insert" in config["init_parameters"]
            assert config["init_parameters"]["use_batch_insert"] is False
            
            # Deserialize
            restored = DB2DocumentStore.from_dict(config)
            assert restored.use_batch_insert is False
            
        finally:
            try:
                store._drop_table_if_exists()
            except Exception:
                pass
