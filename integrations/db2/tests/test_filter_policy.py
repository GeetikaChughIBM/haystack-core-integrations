# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore


@pytest.mark.integration
class TestFilterPolicy:
    """Test filter_policy parameter in DB2EmbeddingRetriever."""

    def test_filter_policy_replace(self, document_store: DB2DocumentStore) -> None:
        """Test REPLACE filter policy (runtime filters replace instance filters)."""
        # Write test documents with embeddings
        embedding = [0.1] * document_store.embedding_dimension
        docs = [
            Document(
                content="Python programming", embedding=embedding, meta={"language": "python", "level": "beginner"}
            ),
            Document(
                content="Java programming", embedding=embedding, meta={"language": "java", "level": "intermediate"}
            ),
            Document(content="Advanced Python", embedding=embedding, meta={"language": "python", "level": "advanced"}),
        ]
        document_store.write_documents(docs)

        # Create retriever with instance filters
        retriever = DB2EmbeddingRetriever(
            document_store=document_store,
            filters={"language": "python"},
            filter_policy=FilterPolicy.REPLACE,
        )

        # Run with runtime filters - should REPLACE instance filters
        query_embedding = [0.1] * document_store.embedding_dimension
        result = retriever.run(query_embedding=query_embedding, filters={"language": "java"})

        # Should only return Java document (runtime filter replaced instance filter)
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["language"] == "java"

    def test_filter_policy_merge(self, document_store: DB2DocumentStore) -> None:
        """Test MERGE filter policy (runtime filters merge with instance filters)."""
        # Write test documents with embeddings
        embedding = [0.1] * document_store.embedding_dimension
        docs = [
            Document(
                content="Python programming", embedding=embedding, meta={"language": "python", "level": "beginner"}
            ),
            Document(
                content="Java programming", embedding=embedding, meta={"language": "java", "level": "intermediate"}
            ),
            Document(content="Advanced Python", embedding=embedding, meta={"language": "python", "level": "advanced"}),
        ]
        document_store.write_documents(docs)

        # Create retriever with instance filters
        retriever = DB2EmbeddingRetriever(
            document_store=document_store,
            filters={"language": "python"},
            filter_policy=FilterPolicy.MERGE,
        )

        # Run with runtime filters - should MERGE with instance filters
        query_embedding = [0.1] * document_store.embedding_dimension
        result = retriever.run(query_embedding=query_embedding, filters={"level": "advanced"})

        # Should only return Advanced Python document (both filters applied)
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["language"] == "python"
        assert result["documents"][0].meta["level"] == "advanced"

    def test_filter_policy_string_value(self, document_store: DB2DocumentStore) -> None:
        """Test filter_policy can be specified as string."""
        # Write test documents with embeddings
        embedding = [0.1] * document_store.embedding_dimension
        docs = [
            Document(content="Python programming", embedding=embedding, meta={"language": "python"}),
            Document(content="Java programming", embedding=embedding, meta={"language": "java"}),
        ]
        document_store.write_documents(docs)

        # Create retriever with string filter_policy
        retriever = DB2EmbeddingRetriever(
            document_store=document_store,
            filters={"language": "python"},
            filter_policy="replace",  # String instead of enum
        )

        # Verify it was converted to enum
        assert retriever.filter_policy == FilterPolicy.REPLACE

        # Run with runtime filters
        query_embedding = [0.1] * document_store.embedding_dimension
        result = retriever.run(query_embedding=query_embedding, filters={"language": "java"})

        # Should only return Java document
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["language"] == "java"

    def test_filter_policy_serialization(self, document_store: DB2DocumentStore) -> None:
        """Test filter_policy is properly serialized and deserialized."""
        retriever = DB2EmbeddingRetriever(
            document_store=document_store,
            filters={"language": "python"},
            filter_policy=FilterPolicy.MERGE,
        )

        # Serialize
        retriever_dict = retriever.to_dict()
        assert retriever_dict["init_parameters"]["filter_policy"] == "merge"

        # Deserialize
        new_retriever = DB2EmbeddingRetriever.from_dict(retriever_dict)
        assert new_retriever.filter_policy == FilterPolicy.MERGE

    def test_filter_policy_no_runtime_filters(self, document_store: DB2DocumentStore) -> None:
        """Test that instance filters are used when no runtime filters provided."""
        # Write test documents with embeddings
        embedding = [0.1] * document_store.embedding_dimension
        docs = [
            Document(content="Python programming", embedding=embedding, meta={"language": "python"}),
            Document(content="Java programming", embedding=embedding, meta={"language": "java"}),
        ]
        document_store.write_documents(docs)

        # Create retriever with instance filters
        retriever = DB2EmbeddingRetriever(
            document_store=document_store,
            filters={"language": "python"},
            filter_policy=FilterPolicy.REPLACE,
        )

        # Run without runtime filters
        query_embedding = [0.1] * document_store.embedding_dimension
        result = retriever.run(query_embedding=query_embedding)

        # Should use instance filters
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["language"] == "python"
