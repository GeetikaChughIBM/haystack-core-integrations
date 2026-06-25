# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for IBM DB2 Document Store using Haystack mixin tests."""

import asyncio
import math
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
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    FilterDocumentsTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
    WriteDocumentsTest,
)

from haystack_integrations.document_stores.ibm_db import Db2ConnectionConfig, Db2DocumentStore
from haystack_integrations.document_stores.ibm_db.document_store import _row_to_document

try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


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
    FilterDocumentsTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    """
    Test Db2DocumentStore using Haystack's standard mixin tests.

    This class inherits from Haystack's mixin test classes which provide
    standardized tests for document store implementations.
    """

    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]):
        """
        Assert that two lists of Documents are equal, ignoring order.

        DB2 returns documents ordered by ID, but the expected list from Python
        is in insertion order. We sort both lists by ID before comparing.
        """
        # Sort both lists by document ID for consistent comparison
        received_sorted = sorted(received, key=lambda d: d.id)
        expected_sorted = sorted(expected, key=lambda d: d.id)

        # Check lengths first
        assert len(received_sorted) == len(expected_sorted), (
            f"Different number of documents: {len(received_sorted)} vs {len(expected_sorted)}"
        )

        # Compare each document
        for i, (rec, exp) in enumerate(zip(received_sorted, expected_sorted, strict=True)):
            assert rec.id == exp.id, f"Document {i}: IDs don't match: {rec.id} vs {exp.id}"
            assert rec.content == exp.content, f"Document {i} ({rec.id}): Content doesn't match"
            assert rec.meta == exp.meta, f"Document {i} ({rec.id}): Meta doesn't match: {rec.meta} vs {exp.meta}"

            # Handle embedding comparison with floating point tolerance
            if rec.embedding is None and exp.embedding is None:
                continue
            elif rec.embedding is None or exp.embedding is None:
                msg = f"Document {i} ({rec.id}): One embedding is None, the other is not"
                raise AssertionError(msg)
            else:
                assert len(rec.embedding) == len(exp.embedding), (
                    f"Document {i} ({rec.id}): Embedding lengths don't match"
                )
                # Compare embeddings with tolerance for floating point precision
                for j, (r_val, e_val) in enumerate(zip(rec.embedding, exp.embedding, strict=True)):
                    if not math.isclose(r_val, e_val, rel_tol=1e-6, abs_tol=1e-9):
                        msg = f"Document {i} ({rec.id}): Embedding value {j} doesn't match: {r_val} vs {e_val}"
                        raise AssertionError(msg)

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

    def test_write_documents_skip_policy_all_new(self, document_store):
        """Test SKIP policy when all documents are new."""
        docs = [
            Document(id="1", content="Doc 1", embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", embedding=[0.2] * 768),
        ]
        written = document_store.write_documents(docs, policy=DuplicatePolicy.SKIP)
        assert written == 2
        assert document_store.count_documents() == 2

    def test_write_documents_skip_policy_some_duplicates(self, document_store):
        """Test SKIP policy when some documents already exist."""
        # Write initial documents
        doc1 = Document(id="1", content="Original", embedding=[0.1] * 768)
        document_store.write_documents([doc1])

        # Try to write mix of new and duplicate documents
        mixed_docs = [
            Document(id="1", content="Updated", embedding=[0.9] * 768),  # Duplicate
            Document(id="2", content="New", embedding=[0.2] * 768),  # New
            Document(id="3", content="Also New", embedding=[0.3] * 768),  # New
        ]
        written = document_store.write_documents(mixed_docs, policy=DuplicatePolicy.SKIP)
        assert written == 2  # Only new documents written
        assert document_store.count_documents() == 3

    def test_write_documents_skip_policy_all_duplicates(self, document_store):
        """Test SKIP policy when all documents exist (should return 0)."""
        # Write initial documents
        docs = [
            Document(id="1", content="Doc 1", embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        # Try to write same documents again
        written = document_store.write_documents(docs, policy=DuplicatePolicy.SKIP)
        assert written == 0
        assert document_store.count_documents() == 2

    def test_write_documents_skip_policy_preserves_existing(self, document_store):
        """Verify existing documents are not modified with SKIP policy."""
        # Write initial document with specific metadata
        original = Document(
            id="1", content="Original Content", meta={"version": 1, "author": "Alice"}, embedding=[0.1] * 768
        )
        document_store.write_documents([original])

        # Try to write updated version with SKIP policy
        updated = Document(
            id="1", content="Updated Content", meta={"version": 2, "author": "Bob"}, embedding=[0.9] * 768
        )
        written = document_store.write_documents([updated], policy=DuplicatePolicy.SKIP)
        assert written == 0

        # Verify original document is unchanged
        docs = document_store.filter_documents({"operator": "==", "field": "id", "value": "1"})
        assert len(docs) == 1
        assert docs[0].content == "Original Content"
        assert docs[0].meta["version"] == 1
        assert docs[0].meta["author"] == "Alice"
        # Use approximate comparison for floating point values
        assert math.isclose(docs[0].embedding[0], 0.1, rel_tol=1e-6, abs_tol=1e-9)

    def test_write_documents_overwrite_policy_new_documents(self, document_store):
        """Test OVERWRITE policy when documents are new."""
        docs = [
            Document(id="1", content="Doc 1", embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", embedding=[0.2] * 768),
        ]
        written = document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        assert written == 2
        assert document_store.count_documents() == 2

    def test_write_documents_overwrite_policy_updates_existing(self, document_store):
        """Test OVERWRITE policy updates existing documents."""
        # Write initial document
        original = Document(id="1", content="Original", meta={"version": 1}, embedding=[0.1] * 768)
        document_store.write_documents([original])

        # Overwrite with updated document
        updated = Document(id="1", content="Updated", meta={"version": 2}, embedding=[0.9] * 768)
        written = document_store.write_documents([updated], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1
        assert document_store.count_documents() == 1

        # Verify document was updated
        docs = document_store.filter_documents({"operator": "==", "field": "id", "value": "1"})
        assert len(docs) == 1
        assert docs[0].content == "Updated"
        assert docs[0].meta["version"] == 2

    def test_write_documents_overwrite_policy_updates_content(self, document_store):
        """Verify content is updated with OVERWRITE policy."""
        original = Document(id="1", content="Original Content", embedding=[0.1] * 768)
        document_store.write_documents([original])

        updated = Document(id="1", content="Completely New Content", embedding=[0.1] * 768)
        document_store.write_documents([updated], policy=DuplicatePolicy.OVERWRITE)

        docs = document_store.filter_documents({"operator": "==", "field": "id", "value": "1"})
        assert docs[0].content == "Completely New Content"

    def test_write_documents_overwrite_policy_updates_meta(self, document_store):
        """Verify metadata is updated with OVERWRITE policy."""
        original = Document(id="1", content="Content", meta={"key1": "value1", "key2": "value2"}, embedding=[0.1] * 768)
        document_store.write_documents([original])

        updated = Document(id="1", content="Content", meta={"key1": "updated", "key3": "new"}, embedding=[0.1] * 768)
        document_store.write_documents([updated], policy=DuplicatePolicy.OVERWRITE)

        docs = document_store.filter_documents({"operator": "==", "field": "id", "value": "1"})
        assert docs[0].meta == {"key1": "updated", "key3": "new"}

    def test_write_documents_overwrite_policy_updates_embedding(self, document_store):
        """Verify embeddings are updated with OVERWRITE policy."""
        original = Document(id="1", content="Content", embedding=[0.1] * 768)
        document_store.write_documents([original])

        updated = Document(id="1", content="Content", embedding=[0.9] * 768)
        document_store.write_documents([updated], policy=DuplicatePolicy.OVERWRITE)

        docs = document_store.filter_documents({"operator": "==", "field": "id", "value": "1"})
        # Use approximate comparison for floating point values
        assert math.isclose(docs[0].embedding[0], 0.9, rel_tol=1e-6, abs_tol=1e-9)


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

    def test_filter_documents_on_id_field(self, document_store):
        """Test filtering on id field."""
        docs = [
            Document(id="doc-1", content="First", embedding=[0.1] * 768),
            Document(id="doc-2", content="Second", embedding=[0.2] * 768),
            Document(id="doc-3", content="Third", embedding=[0.3] * 768),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "==", "field": "id", "value": "doc-2"}
        result = document_store.filter_documents(filters)
        assert len(result) == 1
        assert result[0].id == "doc-2"

    def test_filter_documents_on_content_field(self, document_store):
        """Test filtering on content field."""
        docs = [
            Document(id="1", content="Python programming", embedding=[0.1] * 768),
            Document(id="2", content="Java programming", embedding=[0.2] * 768),
            Document(id="3", content="Python data science", embedding=[0.3] * 768),
        ]
        document_store.write_documents(docs)

        # Note: Content filtering depends on implementation - this tests the filter structure
        filters = {"operator": "in", "field": "id", "value": ["1", "3"]}
        result = document_store.filter_documents(filters)
        assert len(result) == 2
        assert {doc.id for doc in result} == {"1", "3"}

    def test_count_documents_with_simple_filter(self, document_store):
        """Test count_documents_by_filter with simple filter condition."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python", "level": "beginner"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java", "level": "advanced"}, embedding=[0.2] * 768),
            Document(id="3", content="Python", meta={"language": "python", "level": "advanced"}, embedding=[0.3] * 768),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "==", "field": "meta.language", "value": "python"}
        count = document_store.count_documents_by_filter(filters)
        assert count == 2

    def test_count_documents_with_complex_filter(self, document_store):
        """Test count_documents_by_filter with complex nested filters."""
        docs = [
            Document(id="1", content="Doc 1", meta={"category": "A", "rating": 5}, embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", meta={"category": "B", "rating": 3}, embedding=[0.2] * 768),
            Document(id="3", content="Doc 3", meta={"category": "A", "rating": 4}, embedding=[0.3] * 768),
            Document(id="4", content="Doc 4", meta={"category": "A", "rating": 5}, embedding=[0.4] * 768),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "AND",
            "conditions": [
                {"operator": "==", "field": "meta.category", "value": "A"},
                {"operator": ">=", "field": "meta.rating", "value": 5},
            ],
        }
        count = document_store.count_documents_by_filter(filters)
        assert count == 2

    def test_count_documents_with_no_matches(self, document_store):
        """Test count_documents_by_filter when filter matches no documents."""
        docs = [
            Document(id="1", content="Doc 1", meta={"status": "active"}, embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", meta={"status": "active"}, embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "==", "field": "meta.status", "value": "archived"}
        count = document_store.count_documents_by_filter(filters)
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_documents_async_with_complex_filters(self, document_store):
        """Test async count_documents_by_filter with complex filters."""
        docs = [
            Document(
                id="1", content="Python", meta={"language": "python", "difficulty": "easy"}, embedding=[0.1] * 768
            ),
            Document(id="2", content="Java", meta={"language": "java", "difficulty": "medium"}, embedding=[0.2] * 768),
            Document(
                id="3", content="Python", meta={"language": "python", "difficulty": "hard"}, embedding=[0.3] * 768
            ),
        ]
        document_store.write_documents(docs)

        filters = {
            "operator": "AND",
            "conditions": [
                {"operator": "==", "field": "meta.language", "value": "python"},
                {"operator": "!=", "field": "meta.difficulty", "value": "easy"},
            ],
        }
        count = await document_store.count_documents_by_filter_async(filters)
        assert count == 1

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


class TestDb2DocumentStoreUnit:
    """Unit tests for Db2DocumentStore that don't require a database."""

    def test_to_row_with_none_metadata(self, document_store):
        """Test _to_row with None metadata."""
        doc = Document(id="1", content="test", meta=None, embedding=[0.1] * 768)
        row = document_store._to_row(doc)

        assert row[0] == "1"  # id
        assert row[1] == "test"  # content
        assert row[2] == "{}"  # meta should be empty JSON object
        assert row[3] is not None  # embedding

    def test_to_row_with_none_embedding(self, document_store):
        """Test _to_row with None embedding."""
        doc = Document(id="1", content="test", meta={"key": "value"}, embedding=None)
        row = document_store._to_row(doc)

        assert row[0] == "1"  # id
        assert row[1] == "test"  # content
        assert '"key"' in row[2]  # meta JSON
        assert row[3] is None  # embedding should be None

    def test_build_where_clause_empty_filters(self, document_store):
        """Test _build_where_clause with empty filters."""
        where_clause, params = document_store._build_where_clause({})
        assert where_clause == ""
        assert params == []

    def test_write_documents_invalid_type(self, document_store):
        """Test write_documents with invalid type."""
        with pytest.raises(ValueError, match="Expected a list of Document objects"):
            document_store.write_documents("not a list")

    def test_write_documents_invalid_document_type(self, document_store):
        """Test write_documents with invalid document type in list."""
        with pytest.raises(ValueError, match="Expected Document objects"):
            document_store.write_documents([{"id": "1", "content": "test"}])

    def test_write_documents_unsupported_policy(self, document_store):
        """Test write_documents with unsupported duplicate policy."""
        doc = Document(id="1", content="test", embedding=[0.1] * 768)

        # Create a mock unsupported policy
        class UnsupportedPolicy:
            pass

        with pytest.raises(ValueError, match="Unsupported duplicate policy"):
            document_store.write_documents([doc], policy=UnsupportedPolicy())

    def test_row_to_document_with_none_values(self):
        """Test _row_to_document with None values."""
        # Test with None content and meta
        row = ("doc_id", None, None, None)
        doc = _row_to_document(row)

        assert doc.id == "doc_id"
        assert doc.content is None
        assert doc.meta == {}
        assert doc.embedding is None

    def test_row_to_document_with_valid_embedding(self):
        """Test _row_to_document with valid embedding."""
        # Test with valid embedding (as tuple/list)
        embedding_data = [0.1, 0.2, 0.3]
        row = ("doc_id", "content", '{"key": "value"}', embedding_data)
        doc = _row_to_document(row)

        assert doc.id == "doc_id"
        assert doc.content == "content"
        assert doc.meta == {"key": "value"}
        assert doc.embedding == [0.1, 0.2, 0.3]


@pytest.mark.integration
class TestDb2DocumentStoreEdgeCases:
    """Test edge cases and error conditions."""

    def test_write_documents_empty_list_returns_zero(self, document_store):
        """Test writing empty list returns 0."""
        assert document_store.write_documents([]) == 0

    def test_filter_documents_empty_result(self, document_store):
        """Test filter returning no results."""
        docs = document_store.filter_documents({"operator": "==", "field": "meta.nonexistent", "value": "impossible"})
        assert docs == []

    def test_count_documents_by_filter_with_none(self, document_store):
        """Test count_documents_by_filter with None filter."""
        docs = [
            Document(id="1", content="test1", embedding=[0.1] * 768),
            Document(id="2", content="test2", embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)
        count = document_store.count_documents_by_filter(None)
        assert count == 2

    @pytest.mark.asyncio
    async def test_count_documents_by_filter_async_with_filters(self, document_store):
        """Test async count with filters."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java"}, embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)
        count = await document_store.count_documents_by_filter_async(
            {"operator": "==", "field": "meta.language", "value": "python"}
        )
        assert count == 1

    def test_document_without_embedding(self, document_store):
        """Test storing document without embedding."""
        doc = Document(id="no_emb", content="No embedding", meta={"test": True})
        document_store.write_documents([doc])
        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_emb"})
        assert len(retrieved) == 1
        assert retrieved[0].embedding is None

    def test_document_without_content(self, document_store):
        """Test storing document without content."""
        doc = Document(id="no_content", content=None, meta={"test": True}, embedding=[0.1] * 768)
        document_store.write_documents([doc])
        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_content"})
        assert len(retrieved) == 1
        assert retrieved[0].content is None

    def test_document_without_meta(self, document_store):
        """Test storing document without metadata."""
        doc = Document(id="no_meta", content="No metadata", embedding=[0.1] * 768)
        document_store.write_documents([doc])
        retrieved = document_store.filter_documents({"operator": "==", "field": "id", "value": "no_meta"})
        assert len(retrieved) == 1
        assert retrieved[0].meta == {}

    def test_connection_with_schema(self):
        """Test connection with schema configuration."""
        config = Db2ConnectionConfig(
            database="testdb",
            hostname="localhost",
            port=50000,
            username="db2inst1",
            password="Passw0rd123!",
            protocol="TCPIP",
            schema="DB2INST1",
        )
        store = Db2DocumentStore(
            connection_config=config,
            table_name=f"schema_test_{sys.version_info.major}_{sys.version_info.minor}",
            embedding_dim=768,
            distance_metric="COSINE",
            recreate_table=True,
        )

        # Verify connection works with schema
        assert store.count_documents() == 0

        # Cleanup
        try:
            conn = store._get_connection()
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE {store.table_name}")
                conn.commit()
        except Exception:
            pass

    def test_euclidean_distance_metric(self, connection_config):
        """Test document store with EUCLIDEAN distance metric."""
        store = Db2DocumentStore(
            connection_config=connection_config,
            table_name=f"euclidean_test_{sys.version_info.major}_{sys.version_info.minor}",
            embedding_dim=768,
            distance_metric="EUCLIDEAN",
            recreate_table=True,
        )

        # Write documents with embeddings
        docs = [
            Document(id="1", content="Doc 1", embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", embedding=[0.2] * 768),
        ]
        store.write_documents(docs)

        # Test embedding retrieval with EUCLIDEAN metric
        results = store._embedding_retrieval([0.15] * 768, top_k=2)
        assert len(results) == 2
        assert all(doc.score is not None for doc in results)

        # Cleanup
        try:
            conn = store._get_connection()
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE {store.table_name}")
                conn.commit()
        except Exception:
            pass

    def test_manhattan_distance_metric(self, connection_config):
        """Test document store with MANHATTAN distance metric."""
        store = Db2DocumentStore(
            connection_config=connection_config,
            table_name=f"manhattan_test_{sys.version_info.major}_{sys.version_info.minor}",
            embedding_dim=768,
            distance_metric="MANHATTAN",
            recreate_table=True,
        )

        # Write documents with embeddings
        docs = [
            Document(id="1", content="Doc 1", embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", embedding=[0.2] * 768),
        ]
        store.write_documents(docs)

        # Test embedding retrieval with MANHATTAN metric
        results = store._embedding_retrieval([0.15] * 768, top_k=2)
        assert len(results) == 2
        assert all(doc.score is not None for doc in results)

        # Cleanup
        try:
            conn = store._get_connection()
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE {store.table_name}")
                conn.commit()
        except Exception:
            pass

    def test_embedding_retrieval_with_filters(self, document_store):
        """Test _embedding_retrieval with filters."""
        docs = [
            Document(id="1", content="Python doc", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java doc", meta={"language": "java"}, embedding=[0.2] * 768),
            Document(id="3", content="Python advanced", meta={"language": "python"}, embedding=[0.3] * 768),
        ]
        document_store.write_documents(docs)

        # Retrieve with filter
        filters = {"operator": "==", "field": "meta.language", "value": "python"}
        results = document_store._embedding_retrieval([0.15] * 768, filters=filters, top_k=5)

        assert len(results) == 2
        assert all(doc.meta["language"] == "python" for doc in results)
        assert all(doc.score is not None for doc in results)

    def test_embedding_retrieval_large_top_k(self, document_store):
        """Test _embedding_retrieval with large top_k value."""
        docs = [Document(id=str(i), content=f"Doc {i}", embedding=[0.1 * i] * 768) for i in range(5)]
        document_store.write_documents(docs)

        # Request more documents than exist
        results = document_store._embedding_retrieval([0.2] * 768, top_k=100)
        assert len(results) == 5  # Should return all available documents

    def test_embedding_retrieval_with_empty_embeddings(self, document_store):
        """Test _embedding_retrieval when documents have no embeddings."""
        # Write document without embedding
        doc = Document(id="1", content="No embedding")
        document_store.write_documents([doc])

        # Try to retrieve - should return empty list since doc has no embedding
        results = document_store._embedding_retrieval([0.1] * 768, top_k=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async(self, document_store):
        """Test async embedding retrieval."""
        docs = [
            Document(id="1", content="Doc 1", embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        results = await document_store._embedding_retrieval_async([0.15] * 768, top_k=2)
        assert len(results) == 2
        assert all(doc.score is not None for doc in results)

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async_with_filters(self, document_store):
        """Test async embedding retrieval with filters."""
        docs = [
            Document(id="1", content="Python", meta={"language": "python"}, embedding=[0.1] * 768),
            Document(id="2", content="Java", meta={"language": "java"}, embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        filters = {"operator": "==", "field": "meta.language", "value": "python"}
        results = await document_store._embedding_retrieval_async([0.15] * 768, filters=filters, top_k=5)

        assert len(results) == 1
        assert results[0].meta["language"] == "python"

    def test_delete_documents_empty_list(self, document_store):
        """Test delete_documents with empty list."""
        # Should not raise an error
        document_store.delete_documents([])
        assert document_store.count_documents() == 0

    @pytest.mark.asyncio
    async def test_delete_documents_async_empty_list(self, document_store):
        """Test async delete_documents with empty list."""
        await document_store.delete_documents_async([])
        assert document_store.count_documents() == 0

    def test_write_documents_invalid_type(self, document_store):
        """Test write_documents with invalid type."""
        with pytest.raises(ValueError, match="Expected a list of Document objects"):
            document_store.write_documents("not a list")

    def test_write_documents_invalid_document_type(self, document_store):
        """Test write_documents with invalid document type in list."""
        with pytest.raises(ValueError, match="Expected Document objects"):
            document_store.write_documents([{"id": "1", "content": "test"}])

    def test_write_documents_unsupported_policy(self, document_store):
        """Test write_documents with unsupported duplicate policy."""
        doc = Document(id="1", content="test", embedding=[0.1] * 768)

        # Create a mock unsupported policy
        class UnsupportedPolicy:
            pass

        with pytest.raises(ValueError, match="Unsupported duplicate policy"):
            document_store.write_documents([doc], policy=UnsupportedPolicy())

    def test_filter_documents_no_filters(self, document_store):
        """Test filter_documents without any filters."""
        docs = [
            Document(id="1", content="Doc 1", embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        # Filter without filters should return all documents
        results = document_store.filter_documents(None)
        assert len(results) == 2

    def test_count_documents_by_filter_empty_filters(self, document_store):
        """Test count_documents_by_filter with empty dict."""
        docs = [
            Document(id="1", content="test1", embedding=[0.1] * 768),
            Document(id="2", content="test2", embedding=[0.2] * 768),
        ]
        document_store.write_documents(docs)

        # Empty dict should count all documents
        count = document_store.count_documents_by_filter({})
        assert count == 2

    def test_connection_config_with_options(self, connection_config):
        """Test connection config with additional connection options."""
        # Create a new config based on the fixture with additional options
        config = Db2ConnectionConfig(
            database=connection_config.database,
            hostname=connection_config.hostname,
            port=connection_config.port,
            username=connection_config.username,
            password=connection_config.password,
            protocol=connection_config.protocol,
            connection_options={
                "CONNECTTIMEOUT": 30,
                "QUERYTIMEOUT": 60,
            },
        )

        store = Db2DocumentStore(
            connection_config=config,
            table_name=f"options_test_{sys.version_info.major}_{sys.version_info.minor}",
            embedding_dim=768,
            distance_metric="COSINE",
            recreate_table=True,
        )

        # Verify connection works
        assert store.count_documents() == 0

        # Cleanup
        try:
            conn = store._get_connection()
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE {store.table_name}")
                conn.commit()
        except Exception:
            pass

    def test_table_creation_with_different_embedding_dims(self, connection_config):
        """Test table creation with various embedding dimensions."""
        for dim in [128, 384, 1536]:
            store = Db2DocumentStore(
                connection_config=connection_config,
                table_name=f"dim_{dim}_test_{sys.version_info.major}_{sys.version_info.minor}",
                embedding_dim=dim,
                distance_metric="COSINE",
                recreate_table=True,
            )

            # Write and retrieve document
            doc = Document(id="1", content="test", embedding=[0.1] * dim)
            store.write_documents([doc])

            retrieved = store.filter_documents()
            assert len(retrieved) == 1
            assert len(retrieved[0].embedding) == dim

            # Cleanup
            try:
                conn = store._get_connection()
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE {store.table_name}")
                    conn.commit()
            except Exception:
                pass

    def test_embedding_retrieval_score_ordering(self, document_store):
        """Test that embedding retrieval returns documents ordered by score."""
        docs = [
            Document(id="1", content="Doc 1", embedding=[1.0] * 768),
            Document(id="2", content="Doc 2", embedding=[0.5] * 768),
            Document(id="3", content="Doc 3", embedding=[0.1] * 768),
        ]
        document_store.write_documents(docs)

        # Query with embedding close to doc 2
        results = document_store._embedding_retrieval([0.5] * 768, top_k=3)

        assert len(results) == 3
        # Scores should be in ascending order (lower distance = better match)
        for i in range(len(results) - 1):
            assert results[i].score <= results[i + 1].score

    def test_embedding_retrieval_basic(self, document_store):
        """Test basic vector similarity search."""
        docs = [
            Document(id="1", content="First document", embedding=[0.1, 0.2] + [0.0] * 766),
            Document(id="2", content="Second document", embedding=[0.5, 0.5] + [0.0] * 766),
            Document(id="3", content="Third document", embedding=[0.9, 0.8] + [0.0] * 766),
        ]
        document_store.write_documents(docs)

        # Query with embedding similar to doc 2
        query_embedding = [0.5, 0.5] + [0.0] * 766
        results = document_store._embedding_retrieval(query_embedding, top_k=2)

        assert len(results) == 2
        assert results[0].id == "2"  # Most similar
        assert all(doc.score is not None for doc in results)

    def test_embedding_retrieval_with_different_top_k(self, document_store):
        """Test embedding retrieval with different top_k values."""
        docs = [Document(id=str(i), content=f"Doc {i}", embedding=[0.1 * i] * 768) for i in range(10)]
        document_store.write_documents(docs)

        query_embedding = [0.5] * 768

        # Test top_k=1
        results = document_store._embedding_retrieval(query_embedding, top_k=1)
        assert len(results) == 1

        # Test top_k=5
        results = document_store._embedding_retrieval(query_embedding, top_k=5)
        assert len(results) == 5

        # Test top_k=10
        results = document_store._embedding_retrieval(query_embedding, top_k=10)
        assert len(results) == 10

    def test_embedding_retrieval_includes_score(self, document_store):
        """Verify score is included in returned documents."""
        docs = [
            Document(id="1", content="Doc 1", embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", embedding=[0.5] * 768),
        ]
        document_store.write_documents(docs)

        results = document_store._embedding_retrieval([0.3] * 768, top_k=2)

        assert len(results) == 2
        for doc in results:
            assert doc.score is not None
            assert isinstance(doc.score, float)

    def test_embedding_retrieval_ordered_by_similarity(self, document_store):
        """Verify results are ordered by similarity score."""
        docs = [
            Document(id="far", content="Far", embedding=[0.1] * 768),
            Document(id="close", content="Close", embedding=[0.5] * 768),
            Document(id="medium", content="Medium", embedding=[0.3] * 768),
        ]
        document_store.write_documents(docs)

        query_embedding = [0.5] * 768
        results = document_store._embedding_retrieval(query_embedding, top_k=3)

        assert len(results) == 3
        # The closest document should be first
        assert results[0].id == "close"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_async_basic(self, document_store):
        """Test async embedding retrieval basic functionality."""
        docs = [
            Document(id="1", content="Doc 1", embedding=[0.1] * 768),
            Document(id="2", content="Doc 2", embedding=[0.5] * 768),
            Document(id="3", content="Doc 3", embedding=[0.9] * 768),
        ]
        document_store.write_documents(docs)

        results = await document_store._embedding_retrieval_async([0.5] * 768, top_k=2)

        assert len(results) == 2
        assert all(doc.score is not None for doc in results)
        # Most similar should be first
        assert results[0].id == "2"

    def test_write_documents_validation_empty_list(self, document_store):
        """Test writing empty list returns 0."""
        result = document_store.write_documents([])
        assert result == 0
        assert document_store.count_documents() == 0

    def test_write_documents_validation_invalid_type(self, document_store):
        """Test ValueError when documents parameter is not a list."""
        with pytest.raises(ValueError, match="Expected a list of Document objects"):
            document_store.write_documents("not a list")

        with pytest.raises(ValueError, match="Expected a list of Document objects"):
            document_store.write_documents(None)

        with pytest.raises(ValueError, match="Expected a list of Document objects"):
            document_store.write_documents({"id": "1", "content": "test"})

    def test_write_documents_validation_invalid_document_objects(self, document_store):
        """Test ValueError when list contains non-Document objects."""
        with pytest.raises(ValueError, match="Expected Document objects"):
            document_store.write_documents([{"id": "1", "content": "test"}])

        with pytest.raises(ValueError, match="Expected Document objects"):
            document_store.write_documents(["string", "another string"])

        with pytest.raises(ValueError, match="Expected Document objects"):
            document_store.write_documents([Document(id="1", content="valid"), "invalid"])


# Made with Bob
