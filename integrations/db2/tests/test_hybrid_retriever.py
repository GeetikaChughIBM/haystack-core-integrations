# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for DB2HybridRetriever component."""

import pytest
from haystack import Document
from haystack.document_stores.errors import DocumentStoreError

from haystack_integrations.components.retrievers.db2 import DB2HybridRetriever

pytestmark = pytest.mark.integration


class TestDB2HybridRetriever:
    """Test suite for DB2HybridRetriever."""

    def test_init_default(self, document_store_local):
        """Test initialization with default parameters."""
        retriever = DB2HybridRetriever(document_store=document_store_local)

        assert retriever.document_store == document_store_local
        assert retriever.top_k == 10
        assert retriever.embedding_weight == 0.5
        assert retriever.keyword_weight == 0.5
        assert retriever.rrf_k == 60

    def test_init_custom_weights(self, document_store_local):
        """Test initialization with custom weights."""
        retriever = DB2HybridRetriever(
            document_store=document_store_local,
            embedding_weight=0.7,
            keyword_weight=0.3,
            rrf_k=100,
        )

        assert retriever.embedding_weight == 0.7
        assert retriever.keyword_weight == 0.3
        assert retriever.rrf_k == 100

    def test_init_invalid_weights(self, document_store_local):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError, match="embedding_weight must be between"):
            DB2HybridRetriever(
                document_store=document_store_local,
                embedding_weight=-0.5,
                keyword_weight=0.5,
            )

        with pytest.raises(ValueError, match="keyword_weight must be between"):
            DB2HybridRetriever(
                document_store=document_store_local,
                embedding_weight=0.5,
                keyword_weight=-0.3,
            )

    def test_reciprocal_rank_fusion_basic(self, document_store_local):
        """Test RRF algorithm with basic input."""
        retriever = DB2HybridRetriever(document_store=document_store_local, rrf_k=60)

        # Create test documents with scores
        emb_docs = [
            Document(id="doc1", content="test1", score=0.9),
            Document(id="doc2", content="test2", score=0.8),
            Document(id="doc3", content="test3", score=0.7),
        ]

        kw_docs = [
            Document(id="doc2", content="test2", score=0.95),
            Document(id="doc1", content="test1", score=0.85),
            Document(id="doc4", content="test4", score=0.75),
        ]

        # Test RRF fusion
        fused = retriever._reciprocal_rank_fusion(emb_docs, kw_docs, top_k=3)

        # Verify results
        assert len(fused) <= 3
        assert all(isinstance(doc, Document) for doc in fused)

        # doc1 and doc2 should rank higher as they appear in both lists
        doc_ids = [doc.id for doc in fused]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_reciprocal_rank_fusion_empty_lists(self, document_store_local):
        """Test RRF with empty input lists."""
        retriever = DB2HybridRetriever(document_store=document_store_local)

        # Test with empty lists
        result = retriever._reciprocal_rank_fusion([], [], top_k=10)
        assert result == []

        # Test with one empty list
        docs = [Document(id="doc1", content="test", score=0.9)]
        result = retriever._reciprocal_rank_fusion(docs, [], top_k=10)
        assert len(result) == 1
        assert result[0].id == "doc1"

    def test_to_dict(self, document_store_local):
        """Test serialization to dict."""
        retriever = DB2HybridRetriever(
            document_store=document_store_local,
            top_k=5,
            embedding_weight=0.6,
            keyword_weight=0.4,
            rrf_k=100,
        )

        data = retriever.to_dict()

        assert data["type"] == "haystack_integrations.components.retrievers.db2.hybrid_retriever.DB2HybridRetriever"
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["embedding_weight"] == 0.6
        assert data["init_parameters"]["keyword_weight"] == 0.4
        assert data["init_parameters"]["rrf_k"] == 100

    def test_from_dict(self, document_store_local):
        """Test deserialization from dict."""
        # First serialize to get proper format
        retriever = DB2HybridRetriever(
            document_store=document_store_local,
            top_k=5,
            embedding_weight=0.6,
            keyword_weight=0.4,
            rrf_k=100,
        )

        data = retriever.to_dict()

        # Now deserialize
        retriever2 = DB2HybridRetriever.from_dict(data)

        assert retriever2.top_k == 5
        assert retriever2.embedding_weight == 0.6
        assert retriever2.keyword_weight == 0.4
        assert retriever2.rrf_k == 100

    def test_error_handling_raises_exception(self, document_store_local):
        """Test that errors are properly raised as DocumentStoreError."""

        retriever = DB2HybridRetriever(document_store=document_store_local)

        # Test with invalid query_embedding dimension (should raise error)
        # Using wrong dimension to trigger an error
        with pytest.raises(DocumentStoreError):
            retriever.run(
                query="test query",
                query_embedding=[0.1] * 100,  # Wrong dimension (should be 384)
            )
