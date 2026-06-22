# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for IBM DB2 Document Store using Haystack mixin tests."""

import asyncio
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    WriteDocumentsTest,
)

from haystack_integrations.document_stores.ibm_db import Db2ConnectionConfig, Db2DocumentStore

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# DB2 connection configuration for docker-compose DB2 instance
DB2_CONFIG = Db2ConnectionConfig(
    database="testdb",
    hostname="localhost",
    port=50000,
    username="db2inst1",
    password="Passw0rd123!",
    protocol="TCPIP",
)


def _generate_self_signed_cert_pem() -> bytes:
    """
    Generate a self-signed SSL certificate for testing.

    :return: Certificate in PEM format as bytes
    """
    if not CRYPTOGRAPHY_AVAILABLE:
        msg = "cryptography library is required to generate SSL certificates"
        raise ImportError(msg)

    # Generate private key
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

    # Create certificate
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Test"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Test"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Org"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256(), default_backend())
    )

    # Return certificate in PEM format
    return cert.public_bytes(serialization.Encoding.PEM)


@pytest.mark.integration
class TestDocumentStore(
    CountDocumentsTest,
    WriteDocumentsTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    CountDocumentsByFilterTest,
):
    """
    Test Db2DocumentStore using Haystack's standard mixin tests.

    This class inherits from Haystack's mixin test classes which provide
    standardized tests for document store implementations.
    """

    def test_write_documents(self, document_store: Db2DocumentStore):
        """Test basic write with duplicate handling - default policy is NONE."""
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1
        # Default policy is NONE — a second write of the same doc raises DuplicateDocumentError
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([doc])

    def test_to_dict(self, document_store: Db2DocumentStore):
        """Test serializing document store to dictionary."""
        data = document_store.to_dict()

        assert "type" in data
        assert data["type"] == "haystack_integrations.document_stores.ibm_db.document_store.Db2DocumentStore"
        assert "init_parameters" in data

        init_params = data["init_parameters"]
        assert "connection_config" in init_params
        assert init_params["embedding_dim"] == 768
        assert init_params["distance_metric"] == "COSINE"

    def test_from_dict(self, document_store: Db2DocumentStore):
        """Test deserializing document store from dictionary."""
        data = document_store.to_dict()

        # Create new instance from dict
        new_store = Db2DocumentStore.from_dict(data)

        assert new_store.table_name == document_store.table_name
        assert new_store.embedding_dim == document_store.embedding_dim
        assert new_store.distance_metric == document_store.distance_metric
        assert new_store.connection_config.database == document_store.connection_config.database
        assert new_store.connection_config.hostname == document_store.connection_config.hostname

    def test_connection_reuse(self, document_store: Db2DocumentStore):
        """Test that connection is reused across operations."""
        docs = [Document(id="1", content="test", embedding=[0.1] * 768)]

        # Perform multiple operations
        document_store.write_documents(docs)
        count1 = document_store.count_documents()
        retrieved = document_store.filter_documents()
        count2 = document_store.count_documents()

        assert count1 == count2 == 1
        assert len(retrieved) == 1

        # Connection should be the same instance
        conn1 = document_store._get_connection()
        conn2 = document_store._get_connection()
        assert conn1 is conn2

    def test_document_without_embedding(self, document_store: Db2DocumentStore):
        """Test storing document without embedding."""
        doc = Document(id="no_emb", content="Document without embedding", meta={"test": True})
        document_store.write_documents([doc])

        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_emb"})
        assert len(retrieved) == 1
        assert retrieved[0].embedding is None

    def test_document_without_content(self, document_store: Db2DocumentStore):
        """Test storing document without content."""
        doc = Document(id="no_content", content=None, meta={"test": True}, embedding=[0.1] * 768)
        document_store.write_documents([doc])

        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_content"})
        assert len(retrieved) == 1
        assert retrieved[0].content is None

    def test_complex_metadata(self, document_store: Db2DocumentStore):
        """Test storing document with complex nested metadata."""
        doc = Document(
            id="complex_meta",
            content="Document with complex metadata",
            meta={
                "nested": {"level1": {"level2": {"level3": "deep"}}},
                "list": [1, 2, 3, "four"],
                "mixed": {"numbers": [1, 2, 3], "strings": ["a", "b", "c"]},
            },
            embedding=[0.1] * 768,
        )
        document_store.write_documents([doc])

        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "complex_meta"})
        assert len(retrieved) == 1
        assert retrieved[0].meta["nested"]["level1"]["level2"]["level3"] == "deep"
        assert retrieved[0].meta["list"] == [1, 2, 3, "four"]


@pytest.mark.integration
class TestDb2DocumentStoreDuplicatePolicies:
    """Test duplicate handling policies."""

    def test_duplicate_policy_none(self, document_store):
        """Test NONE policy - fails on duplicates via database integrity errors."""
        doc = Document(id="1", content="test", embedding=[0.1] * 768)
        document_store.write_documents([doc], policy=DuplicatePolicy.NONE)

        # Try to insert duplicate with NONE policy - should raise DuplicateDocumentError
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([doc], policy=DuplicatePolicy.NONE)

    def test_duplicate_policy_fail(self, document_store):
        """Test FAIL policy - fails on duplicates via database integrity errors."""
        doc = Document(id="2", content="test", embedding=[0.1] * 768)
        document_store.write_documents([doc], policy=DuplicatePolicy.FAIL)

        # Try to insert duplicate with FAIL policy - should raise DuplicateDocumentError
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([doc], policy=DuplicatePolicy.FAIL)

    def test_duplicate_policy_skip(self, document_store):
        """Test SKIP policy - skips duplicates."""
        doc1 = Document(id="1", content="Original", meta={"version": 1}, embedding=[0.1] * 768)
        document_store.write_documents([doc1])

        # Try to insert duplicate - should skip it
        doc2 = Document(id="1", content="Updated", meta={"version": 2}, embedding=[0.2] * 768)
        written = document_store.write_documents([doc2], policy=DuplicatePolicy.SKIP)
        assert written == 0  # No documents written

        # Original document should remain unchanged
        docs = document_store.filter_documents({"operator": "==", "field": "id", "value": "1"})
        assert len(docs) == 1
        assert docs[0].content == "Original"
        assert docs[0].meta["version"] == 1

    def test_duplicate_policy_overwrite(self, document_store):
        """Test OVERWRITE policy - updates existing documents."""
        doc1 = Document(id="1", content="Original", meta={"version": 1}, embedding=[0.1] * 768)
        document_store.write_documents([doc1])

        # Update existing document
        doc2 = Document(id="1", content="Updated", meta={"version": 2}, embedding=[0.2] * 768)
        written = document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1

        # Verify document was updated
        docs = document_store.filter_documents({"operator": "==", "field": "id", "value": "1"})
        assert len(docs) == 1
        assert docs[0].content == "Updated"
        assert docs[0].meta["version"] == 2

    def test_duplicate_policy_overwrite_mixed(self, document_store):
        """Test OVERWRITE policy with mix of new and existing documents."""
        doc1 = Document(id="1", content="Original", embedding=[0.1] * 768)
        document_store.write_documents([doc1])

        # Mix of existing and new documents
        mixed_docs = [
            Document(id="1", content="Updated", meta={"updated": True}, embedding=[0.9] * 768),
            Document(id="2", content="New", meta={"new": True}, embedding=[0.4] * 768),
        ]
        written = document_store.write_documents(mixed_docs, policy=DuplicatePolicy.OVERWRITE)
        assert written == 2
        assert document_store.count_documents() == 2


@pytest.mark.integration
class TestDb2DocumentStoreFiltering:
    """Test filtering operations with various operators."""

    def test_filter_equality(self, document_store):
        """Test equality filter."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java"}, embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "==", "field": "meta.language", "value": "python"}
        result = document_store.filter_documents(filters)
        assert len(result) == 1
        assert result[0].id == "1"

    def test_filter_inequality(self, document_store):
        """Test inequality filter."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java"}, embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "!=", "field": "meta.language", "value": "python"}
        result = document_store.filter_documents(filters)
        assert len(result) == 1
        assert result[0].id == "2"

    def test_filter_comparison_operators(self, document_store):
        """Test comparison operators (>, >=, <, <=)."""
        docs = [
            Document(id="1", content="Doc 1", meta={"rating": 3}, embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", meta={"rating": 5}, embedding=[0.2] * 768),
            Document(id="3", content="Doc 3", meta={"rating": 7}, embedding=[0.3] * 768),
        ]
        document_store.write_documents(docs)

        # Test >
        filters = {"operator": ">", "field": "meta.rating", "value": 4}
        result = document_store.filter_documents(filters)
        assert len(result) == 2
        assert {doc.id for doc in result} == {"2", "3"}

        # Test >=
        filters = {"operator": ">=", "field": "meta.rating", "value": 5}
        result = document_store.filter_documents(filters)
        assert len(result) == 2
        assert {doc.id for doc in result} == {"2", "3"}

        # Test <
        filters = {"operator": "<", "field": "meta.rating", "value": 5}
        result = document_store.filter_documents(filters)
        assert len(result) == 1
        assert result[0].id == "1"

        # Test <=
        filters = {"operator": "<=", "field": "meta.rating", "value": 5}
        result = document_store.filter_documents(filters)
        assert len(result) == 2
        assert {doc.id for doc in result} == {"1", "2"}

    def test_filter_in_operator(self, document_store):
        """Test IN operator."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java"}, embedding=[0.2] * 768),
            Document(id="3", content="Go", meta={"language": "go"}, embedding=[0.3] * 768),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "in", "field": "meta.language", "value": ["python", "java"]}
        result = document_store.filter_documents(filters)
        assert len(result) == 2
        assert {doc.id for doc in result} == {"1", "2"}

    def test_filter_not_in_operator(self, document_store):
        """Test NOT IN operator."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java"}, embedding=[0.2] * 768),
            Document(id="3", content="Go", meta={"language": "go"}, embedding=[0.3] * 768),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "not in", "field": "meta.language", "value": ["java"]}
        result = document_store.filter_documents(filters)
        assert len(result) == 2
        assert {doc.id for doc in result} == {"1", "3"}

    def test_filter_and_operator(self, document_store):
        """Test AND logical operator."""
        docs = [
            Document(
                id="1",
                content="Python beginner",
                meta={"language": "python", "level": "beginner"},
                embedding=[0.1] * 768,
            ),
            Document(
                id="2",
                content="Python advanced",
                meta={"language": "python", "level": "advanced"},
                embedding=[0.2] * 768,
            ),
            Document(
                id="3",
                content="Java beginner",
                meta={"language": "java", "level": "beginner"},
                embedding=[0.3] * 768,
            ),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "AND",
            "conditions": [
                {"operator": "==", "field": "meta.language", "value": "python"},
                {"operator": "==", "field": "meta.level", "value": "beginner"},
            ],
        }
        result = document_store.filter_documents(filters)
        assert len(result) == 1
        assert result[0].id == "1"

    def test_filter_or_operator(self, document_store):
        """Test OR logical operator."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java"}, embedding=[0.2] * 768),
            Document(id="3", content="Go", meta={"language": "go"}, embedding=[0.3] * 768),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "OR",
            "conditions": [
                {"operator": "==", "field": "meta.language", "value": "python"},
                {"operator": "==", "field": "meta.language", "value": "java"},
            ],
        }
        result = document_store.filter_documents(filters)
        assert len(result) == 2
        assert {doc.id for doc in result} == {"1", "2"}

    def test_filter_not_operator(self, document_store):
        """Test NOT logical operator."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java"}, embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "NOT",
            "conditions": [
                {"operator": "==", "field": "meta.language", "value": "python"},
            ],
        }
        result = document_store.filter_documents(filters)
        assert len(result) == 1
        assert result[0].id == "2"

    def test_filter_complex_nested(self, document_store):
        """Test complex nested filters."""
        docs = [
            Document(
                id="1",
                content="Python beginner",
                meta={"language": "python", "level": "beginner", "rating": 5},
                embedding=[0.1] * 768,
            ),
            Document(
                id="2",
                content="Python advanced",
                meta={"language": "python", "level": "advanced", "rating": 4},
                embedding=[0.2] * 768,
            ),
            Document(
                id="3",
                content="Java beginner",
                meta={"language": "java", "level": "beginner", "rating": 5},
                embedding=[0.3] * 768,
            ),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "AND",
            "conditions": [
                {
                    "operator": "OR",
                    "conditions": [
                        {"operator": "==", "field": "meta.language", "value": "python"},
                        {"operator": "==", "field": "meta.language", "value": "java"},
                    ],
                },
                {"operator": ">=", "field": "meta.rating", "value": 5},
            ],
        }
        result = document_store.filter_documents(filters)
        assert len(result) == 2
        assert {doc.id for doc in result} == {"1", "3"}


@pytest.mark.integration
class TestDb2DocumentStoreAsync:
    """Test async operations."""

    @pytest.mark.asyncio
    async def test_count_documents_async(self, document_store):
        """Test async document counting."""
        docs = [Document(id="1", content="test", embedding=[0.1] * 768)]
        document_store.write_documents(docs)
        count = await document_store.count_documents_async()
        assert count == 1

    @pytest.mark.asyncio
    async def test_write_documents_async(self, document_store):
        """Test async document writing."""
        docs = [Document(id="1", content="test", embedding=[0.1] * 768)]
        written = await document_store.write_documents_async(docs)
        assert written == 1
        assert document_store.count_documents() == 1

    @pytest.mark.asyncio
    async def test_filter_documents_async(self, document_store):
        """Test async document filtering."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java"}, embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "==", "field": "meta.language", "value": "python"}
        result = await document_store.filter_documents_async(filters)
        assert len(result) == 1
        assert result[0].id == "1"

    @pytest.mark.asyncio
    async def test_delete_documents_async(self, document_store):
        """Test async document deletion."""
        docs = [
            Document(id="1", content="test1", embedding=[0.1] * 768),
            Document(id="2", content="test2", embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)
        await document_store.delete_documents_async(["1"])

        count = await document_store.count_documents_async()
        assert count == 1

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, document_store):
        """Test concurrent async operations."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java"}, embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        # Run multiple operations concurrently
        results = await asyncio.gather(
            document_store.count_documents_async(),
            document_store.filter_documents_async({"operator": "==", "field": "meta.language", "value": "python"}),
            document_store.filter_documents_async({"operator": "==", "field": "meta.language", "value": "java"}),
        )

        count, python_docs, java_docs = results
        assert count == 2
        assert len(python_docs) == 1
        assert len(java_docs) == 1


@pytest.mark.integration
class TestDb2DocumentStoreSSL:
    """Test SSL connection handling."""

    @pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography library not available")
    @pytest.mark.skip(reason="SSL server not available in test environment")
    def test_ssl_connection(self):
        """
        Test that SSL connection works with DB2.

        DB2 provides SSL connections on port 50001 by default.
        The test automatically generates a self-signed certificate for testing purposes.
        Note: IBM DB2 requires the certificate to be in a file.
        """
        # Generate self-signed certificate PEM content
        cert_pem = _generate_self_signed_cert_pem()

        # IBM DB2 requires certificate file path, so write to temporary file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pem", delete=False) as cert_file:
            cert_file.write(cert_pem)
            cert_path = cert_file.name

        try:
            config = Db2ConnectionConfig(
                database="testdb",
                hostname="localhost",
                port=50001,  # SSL port
                username="db2inst1",
                password="Passw0rd123!",
                protocol="TCPIP",
                use_ssl=True,
                ssl_certificate=cert_path,
            )

            store = Db2DocumentStore(
                connection_config=config,
                table_name=f"ssl_test_{sys.version_info.major}_{sys.version_info.minor}",
                embedding_dim=768,
                distance_metric="COSINE",
                recreate_table=True,
            )

            # Verify SSL connection works by executing a query
            assert store.count_documents() == 0

            # Cleanup table
            try:
                conn = store._get_connection()
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE {store.table_name}")
                    conn.commit()
            except Exception:
                pass
        finally:
            # Cleanup certificate file
            Path(cert_path).unlink(missing_ok=True)


# Made with Bob
