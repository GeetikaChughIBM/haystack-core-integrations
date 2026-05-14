# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.document_stores.db2 import DB2DocumentStore


class TestSchemaSupport:
    """Test schema_name support for organizing tables in different schemas."""

    def test_qualified_table_name_with_schema(self):
        """Test that qualified_table_name is correctly built with schema."""
        store = DB2DocumentStore(
            database=Secret.from_env_var("DB2_DATABASE").resolve_value(),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="documents",
            schema_name="myschema",
            embedding_dimension=384,
        )

        assert store.schema_name == "myschema"
        assert store.table_name == "documents"
        assert store.qualified_table_name == "myschema.documents"

    def test_qualified_table_name_without_schema(self):
        """Test that qualified_table_name works without schema."""
        store = DB2DocumentStore(
            database=Secret.from_env_var("DB2_DATABASE").resolve_value(),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="documents",
            embedding_dimension=384,
        )

        assert store.schema_name is None
        assert store.table_name == "documents"
        assert store.qualified_table_name == "documents"

    def test_query_builder_uses_qualified_name(self):
        """Test that QueryBuilder receives the qualified table name."""
        store = DB2DocumentStore(
            database=Secret.from_env_var("DB2_DATABASE").resolve_value(),
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            table_name="docs",
            schema_name="test_schema",
            embedding_dimension=384,
        )

        # QueryBuilder should use the qualified name
        assert store.query_builder.table_name == "test_schema.docs"

    @pytest.mark.integration
    def test_schema_operations(self, document_store_local):
        """Test basic operations with schema support."""
        # Note: This test uses the default schema from conftest
        # In production, users would specify their own schema

        docs = [
            Document(content="First document", embedding=[0.1] * 384),
            Document(content="Second document", embedding=[0.2] * 384),
        ]

        # Write documents
        document_store_local.write_documents(docs)

        # Count documents
        count = document_store_local.count_documents()
        assert count == 2

        # Filter documents
        retrieved = document_store_local.filter_documents()
        assert len(retrieved) == 2

        # Delete documents
        document_store_local.delete_documents([docs[0].id])
        count = document_store_local.count_documents()
        assert count == 1
