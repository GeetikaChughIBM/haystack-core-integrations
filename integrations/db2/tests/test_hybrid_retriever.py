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

    def test_init_default(self, document_store_env):
        """Test initialization with default parameters."""
        retriever = DB2HybridRetriever(document_store=document_store_env)

        assert retriever.document_store == document_store_env
        assert retriever.top_k == 10
        assert retriever.embedding_weight == 0.5
        assert retriever.keyword_weight == 0.5
        assert retriever.rrf_k == 60

    def test_init_custom_weights(self, document_store_env):
        """Test initialization with custom weights."""
        retriever = DB2HybridRetriever(
            document_store=document_store_env,
            embedding_weight=0.7,
            keyword_weight=0.3,
            rrf_k=100,
        )

        assert retriever.embedding_weight == 0.7
        assert retriever.keyword_weight == 0.3
        assert retriever.rrf_k == 100

    def test_init_invalid_weights(self, document_store_env):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError, match="embedding_weight must be between"):
            DB2HybridRetriever(
                document_store=document_store_env,
                embedding_weight=-0.5,
                keyword_weight=0.5,
            )

        with pytest.raises(ValueError, match="keyword_weight must be between"):
            DB2HybridRetriever(
                document_store=document_store_env,
                embedding_weight=0.5,
                keyword_weight=-0.3,
            )

    def test_reciprocal_rank_fusion_basic(self, document_store_env):
        """Test RRF algorithm with basic input."""
        retriever = DB2HybridRetriever(document_store=document_store_env, rrf_k=60)

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

    def test_reciprocal_rank_fusion_empty_lists(self, document_store_env):
        """Test RRF with empty input lists."""
        retriever = DB2HybridRetriever(document_store=document_store_env)

        # Test with empty lists
        result = retriever._reciprocal_rank_fusion([], [], top_k=10)
        assert result == []

        # Test with one empty list
        docs = [Document(id="doc1", content="test", score=0.9)]
        result = retriever._reciprocal_rank_fusion(docs, [], top_k=10)
        assert len(result) == 1
        assert result[0].id == "doc1"

    def test_to_dict(self, document_store_env):
        """Test serialization to dict."""
        retriever = DB2HybridRetriever(
            document_store=document_store_env,
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

    def test_from_dict(self, document_store_env):
        """Test deserialization from dict."""
        # First serialize to get proper format
        retriever = DB2HybridRetriever(
            document_store=document_store_env,
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

    def test_error_handling_raises_exception(self, document_store_env):
        """Test that errors are properly raised as DocumentStoreError."""

        retriever = DB2HybridRetriever(document_store=document_store_env)

        # Test with invalid query_embedding dimension (should raise error)
        # Using wrong dimension to trigger an error
        with pytest.raises(DocumentStoreError):
            retriever.run(
                query="test query",
                query_embedding=[0.1] * 100,  # Wrong dimension (should be 384)
            )


class TestDB2HybridRetrieverEndToEnd:
    """End-to-end tests for hybrid retrieval with real documents."""

    def test_hybrid_retrieval_end_to_end(self, document_store_env):
        """Test complete hybrid retrieval workflow."""
        # Create documents with varied content for testing
        docs = [
            Document(
                id="doc1",
                content="Python machine learning tutorial for beginners",
                embedding=[0.1, 0.2, 0.3] + [0.0] * (document_store_env.embedding_dimension - 3),
                meta={"category": "AI", "difficulty": "beginner"},
            ),
            Document(
                id="doc2",
                content="Advanced deep learning with neural networks",
                embedding=[0.15, 0.25, 0.35] + [0.0] * (document_store_env.embedding_dimension - 3),
                meta={"category": "AI", "difficulty": "advanced"},
            ),
            Document(
                id="doc3",
                content="Python web development with Django framework",
                embedding=[0.5, 0.6, 0.7] + [0.0] * (document_store_env.embedding_dimension - 3),
                meta={"category": "Web", "difficulty": "intermediate"},
            ),
            Document(
                id="doc4",
                content="Machine learning algorithms and Python implementation",
                embedding=[0.12, 0.22, 0.32] + [0.0] * (document_store_env.embedding_dimension - 3),
                meta={"category": "AI", "difficulty": "intermediate"},
            ),
        ]

        document_store_env.write_documents(docs)

        # Create hybrid retriever
        retriever = DB2HybridRetriever(
            document_store=document_store_env,
            embedding_weight=0.6,
            keyword_weight=0.4,
            top_k=3,
        )

        # Query with both text and embedding
        query = "Python machine learning"
        query_embedding = [0.1, 0.2, 0.3] + [0.0] * (document_store_env.embedding_dimension - 3)

        results = retriever.run(query=query, query_embedding=query_embedding)

        # Verify results
        assert "documents" in results
        assert len(results["documents"]) <= 3
        assert all(isinstance(doc, Document) for doc in results["documents"])

        # Verify hybrid scores are present
        for doc in results["documents"]:
            assert doc.score is not None
            assert "_hybrid_score" in doc.meta
            assert "_embedding_score" in doc.meta
            assert "_keyword_score" in doc.meta

        # Documents matching both query terms should rank higher
        doc_ids = [doc.id for doc in results["documents"]]
        # doc1 and doc4 contain both "Python" and "machine learning"
        assert "doc1" in doc_ids or "doc4" in doc_ids

    def test_hybrid_retrieval_with_filters(self, document_store_env):
        """Test hybrid retrieval with metadata filters."""
        docs = [
            Document(
                id="doc1",
                content="Python AI tutorial",
                embedding=[0.1] * document_store_env.embedding_dimension,
                meta={"category": "AI"},
            ),
            Document(
                id="doc2",
                content="Python web tutorial",
                embedding=[0.2] * document_store_env.embedding_dimension,
                meta={"category": "Web"},
            ),
            Document(
                id="doc3",
                content="Python AI guide",
                embedding=[0.15] * document_store_env.embedding_dimension,
                meta={"category": "AI"},
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2HybridRetriever(
            document_store=document_store_env,
            filters={"category": "AI"},
            top_k=10,
        )

        query_embedding = [0.1] * document_store_env.embedding_dimension
        results = retriever.run(query="Python", query_embedding=query_embedding)

        # Should only return AI category documents
        assert len(results["documents"]) == 2
        assert all(doc.meta["category"] == "AI" for doc in results["documents"])

    def test_hybrid_retrieval_weight_impact(self, document_store_env):
        """Test that weights affect ranking."""
        docs = [
            Document(
                id="doc1",
                content="Python Python Python",  # High keyword match
                embedding=[0.5] * document_store_env.embedding_dimension,  # Low embedding similarity
                meta={"type": "keyword_heavy"},
            ),
            Document(
                id="doc2",
                content="Java programming",  # Low keyword match
                embedding=[0.1] * document_store_env.embedding_dimension,  # High embedding similarity
                meta={"type": "embedding_heavy"},
            ),
        ]

        document_store_env.write_documents(docs)

        query_embedding = [0.1] * document_store_env.embedding_dimension

        # Test with keyword-heavy weights
        retriever_kw = DB2HybridRetriever(
            document_store=document_store_env,
            embedding_weight=0.2,
            keyword_weight=0.8,
        )
        results_kw = retriever_kw.run(query="Python", query_embedding=query_embedding)

        # Test with embedding-heavy weights
        retriever_emb = DB2HybridRetriever(
            document_store=document_store_env,
            embedding_weight=0.8,
            keyword_weight=0.2,
        )
        results_emb = retriever_emb.run(query="Python", query_embedding=query_embedding)

        # Verify different rankings based on weights
        assert len(results_kw["documents"]) >= 1
        assert len(results_emb["documents"]) >= 1

        # With keyword-heavy weights, doc1 should rank higher
        # With embedding-heavy weights, doc2 should rank higher
        if len(results_kw["documents"]) >= 2 and len(results_emb["documents"]) >= 2:
            # Rankings should be different
            kw_top = results_kw["documents"][0].id
            emb_top = results_emb["documents"][0].id
            # At least one should prefer doc1 (keyword) and one should prefer doc2 (embedding)
            assert kw_top != emb_top or results_kw["documents"][0].score != results_emb["documents"][0].score


class TestDB2HybridRetrieverMinScore:
    """Test min_score threshold filtering."""

    def test_min_score_filtering(self, document_store_env):
        """Test that min_score filters out low-scoring documents."""
        docs = [
            Document(
                id="doc1",
                content="Python machine learning",
                embedding=[0.1, 0.2, 0.3] + [0.0] * (document_store_env.embedding_dimension - 3),
            ),
            Document(
                id="doc2",
                content="Java programming",
                embedding=[0.8, 0.9, 1.0] + [0.0] * (document_store_env.embedding_dimension - 3),
            ),
            Document(
                id="doc3",
                content="Python tutorial",
                embedding=[0.15, 0.25, 0.35] + [0.0] * (document_store_env.embedding_dimension - 3),
            ),
        ]

        document_store_env.write_documents(docs)

        # Create retriever with min_score threshold
        retriever = DB2HybridRetriever(
            document_store=document_store_env,
            min_score=0.01,  # Set a threshold
            top_k=10,
        )

        query_embedding = [0.1, 0.2, 0.3] + [0.0] * (document_store_env.embedding_dimension - 3)
        results = retriever.run(query="Python", query_embedding=query_embedding)

        # All returned documents should have score >= min_score
        for doc in results["documents"]:
            assert doc.score >= 0.01

    def test_min_score_filters_all_documents(self, document_store_env):
        """Test that very high min_score can filter out all documents."""
        docs = [
            Document(
                id="doc1",
                content="Test document",
                embedding=[0.1] * document_store_env.embedding_dimension,
            ),
        ]

        document_store_env.write_documents(docs)

        # Create retriever with very high min_score
        retriever = DB2HybridRetriever(
            document_store=document_store_env,
            min_score=0.99,  # Very high threshold
            top_k=10,
        )

        query_embedding = [0.5] * document_store_env.embedding_dimension
        results = retriever.run(query="unrelated query", query_embedding=query_embedding)

        # May return no documents if all scores are below threshold
        assert "documents" in results
        assert all(doc.score >= 0.99 for doc in results["documents"])

    def test_min_score_none_returns_all(self, document_store_env):
        """Test that min_score=None returns all documents."""
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Document {i}",
                embedding=[0.1 * i] * document_store_env.embedding_dimension,
            )
            for i in range(5)
        ]

        document_store_env.write_documents(docs)

        # Create retriever without min_score
        retriever = DB2HybridRetriever(
            document_store=document_store_env,
            min_score=None,
            top_k=3,
        )

        query_embedding = [0.1] * document_store_env.embedding_dimension
        results = retriever.run(query="Document", query_embedding=query_embedding)

        # Should return top_k documents regardless of score
        assert len(results["documents"]) == 3


class TestDB2HybridRetrieverRRFDetails:
    """Detailed tests for RRF algorithm behavior."""

    def test_rrf_document_appearing_in_both_lists(self, document_store_env):
        """Test that documents in both lists get boosted scores."""
        retriever = DB2HybridRetriever(
            document_store=document_store_env,
            rrf_k=60,
        )

        # Create documents that appear in both lists
        emb_docs = [
            Document(id="doc1", content="test1", score=0.9),
            Document(id="doc2", content="test2", score=0.8),
        ]

        kw_docs = [
            Document(id="doc1", content="test1", score=0.95),  # Also in embedding results
            Document(id="doc3", content="test3", score=0.85),
        ]

        fused = retriever._reciprocal_rank_fusion(emb_docs, kw_docs, top_k=3)

        # doc1 appears in both lists, so should have highest combined score
        assert fused[0].id == "doc1"
        assert fused[0].meta["_hybrid_score"] > 0

    def test_rrf_k_parameter_effect(self, document_store_env):
        """Test that rrf_k parameter affects scoring."""
        docs_list = [
            Document(id="doc1", content="test1", score=0.9),
            Document(id="doc2", content="test2", score=0.8),
        ]

        # Test with different rrf_k values
        retriever_k60 = DB2HybridRetriever(document_store=document_store_env, rrf_k=60)
        retriever_k100 = DB2HybridRetriever(document_store=document_store_env, rrf_k=100)

        fused_k60 = retriever_k60._reciprocal_rank_fusion(docs_list, [], top_k=2)
        fused_k100 = retriever_k100._reciprocal_rank_fusion(docs_list, [], top_k=2)

        # Different rrf_k should produce different scores
        assert fused_k60[0].score != fused_k100[0].score

    def test_rrf_metadata_enrichment(self, document_store_env):
        """Test that RRF adds metadata to documents."""
        retriever = DB2HybridRetriever(document_store=document_store_env)

        emb_docs = [Document(id="doc1", content="test", score=0.9)]
        kw_docs = [Document(id="doc1", content="test", score=0.8)]

        fused = retriever._reciprocal_rank_fusion(emb_docs, kw_docs, top_k=1)

        # Check metadata enrichment
        assert len(fused) == 1
        doc = fused[0]
        assert "_hybrid_score" in doc.meta
        assert "_embedding_score" in doc.meta
        assert "_keyword_score" in doc.meta
        assert doc.score == doc.meta["_hybrid_score"]
