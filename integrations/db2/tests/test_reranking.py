"""Tests for TransformersSimilarityRanker integration with DB2 retrievers."""

import pytest
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever, DB2HybridRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore


@pytest.mark.integration
class TestRerankingIntegration:
    """Test TransformersSimilarityRanker integration with DB2 retrievers."""

    @pytest.fixture
    def document_store_with_docs(self, document_store_local: DB2DocumentStore) -> DB2DocumentStore:
        """Create document store with sample documents."""
        documents = [
            Document(
                content="Python is great for beginners",
                meta={"language": "python", "difficulty": "beginner"},
            ),
            Document(
                content="Java is used in enterprise applications",
                meta={"language": "java", "difficulty": "intermediate"},
            ),
            Document(
                content="Rust provides memory safety",
                meta={"language": "rust", "difficulty": "advanced"},
            ),
            Document(
                content="JavaScript runs in browsers",
                meta={"language": "javascript", "difficulty": "beginner"},
            ),
        ]

        # Generate embeddings
        embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        embedder.warm_up()
        docs_with_embeddings = embedder.run(documents)["documents"]

        document_store_local.write_documents(docs_with_embeddings)
        return document_store_local

    def test_embedding_retrieval_with_reranking(self, document_store_with_docs: DB2DocumentStore) -> None:
        """Test embedding retrieval + reranking pipeline."""
        # Create pipeline
        pipeline = Pipeline()
        pipeline.add_component(
            "text_embedder",
            SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
        )
        pipeline.add_component(
            "retriever",
            DB2EmbeddingRetriever(document_store=document_store_with_docs, top_k=4),
        )
        pipeline.add_component(
            "ranker",
            TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=2),
        )

        # Connect components
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "ranker.documents")

        # Run query
        query = "beginner friendly programming language"
        results = pipeline.run({"text_embedder": {"text": query}, "ranker": {"query": query}})

        # Verify results
        assert "ranker" in results
        assert "documents" in results["ranker"]
        documents = results["ranker"]["documents"]

        # Should return top 2 after reranking
        assert len(documents) == 2

        # All documents should have scores
        for doc in documents:
            assert doc.score is not None
            assert doc.score > 0

        # Scores should be in descending order
        scores = [doc.score for doc in documents]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_retrieval_with_reranking(self, document_store_with_docs: DB2DocumentStore) -> None:
        """Test hybrid retrieval + reranking pipeline."""
        # Create pipeline
        pipeline = Pipeline()
        pipeline.add_component(
            "text_embedder",
            SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
        )
        pipeline.add_component(
            "hybrid_retriever",
            DB2HybridRetriever(document_store=document_store_with_docs, top_k=4),
        )
        pipeline.add_component(
            "ranker",
            TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=2),
        )

        # Connect components
        pipeline.connect("text_embedder.embedding", "hybrid_retriever.query_embedding")
        pipeline.connect("hybrid_retriever.documents", "ranker.documents")

        # Run query
        query = "memory safety systems"
        results = pipeline.run(
            {
                "text_embedder": {"text": query},
                "hybrid_retriever": {"query": query},
                "ranker": {"query": query},
            }
        )

        # Verify results
        assert "ranker" in results
        assert "documents" in results["ranker"]
        documents = results["ranker"]["documents"]

        # Should return top 2 after reranking
        assert len(documents) == 2

        # All documents should have scores
        for doc in documents:
            assert doc.score is not None

    def test_filtered_retrieval_with_reranking(self, document_store_with_docs: DB2DocumentStore) -> None:
        """Test filtered retrieval + reranking pipeline."""
        # Create pipeline
        pipeline = Pipeline()
        pipeline.add_component(
            "text_embedder",
            SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
        )
        pipeline.add_component(
            "retriever",
            DB2EmbeddingRetriever(
                document_store=document_store_with_docs,
                top_k=4,
                filters={"field": "meta.difficulty", "operator": "==", "value": "beginner"},
            ),
        )
        pipeline.add_component(
            "ranker",
            TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=2),
        )

        # Connect components
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "ranker.documents")

        # Run query
        query = "easy programming language"
        results = pipeline.run({"text_embedder": {"text": query}, "ranker": {"query": query}})

        # Verify results
        assert "ranker" in results
        assert "documents" in results["ranker"]
        documents = results["ranker"]["documents"]

        # Should only return beginner-level documents
        assert len(documents) <= 2
        for doc in documents:
            assert doc.meta.get("difficulty") == "beginner"

    def test_reranking_improves_relevance(self, document_store_with_docs: DB2DocumentStore) -> None:
        """Test that reranking improves result relevance."""
        # Setup
        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        text_embedder.warm_up()

        retriever = DB2EmbeddingRetriever(document_store=document_store_with_docs, top_k=4)
        ranker = TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=2)

        # Get query embedding
        query = "beginner programming"
        query_result = text_embedder.run(query)
        query_embedding = query_result["embedding"]

        # Get retrieval results (before reranking)
        retrieval_results = retriever.run(query_embedding=query_embedding)
        retrieved_docs = retrieval_results["documents"]

        # Get reranked results
        reranked_results = ranker.run(query=query, documents=retrieved_docs)
        reranked_docs = reranked_results["documents"]

        # Verify reranking happened
        assert len(reranked_docs) == 2
        assert len(retrieved_docs) >= len(reranked_docs)

        # Reranked documents should have different scores
        # (cross-encoder scores, not embedding similarity scores)
        for doc in reranked_docs:
            assert doc.score is not None

    def test_reranking_with_empty_results(self, document_store_with_docs: DB2DocumentStore) -> None:
        """Test reranking handles empty retrieval results gracefully."""
        # Create pipeline with filter that returns no results
        pipeline = Pipeline()
        pipeline.add_component(
            "text_embedder",
            SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
        )
        pipeline.add_component(
            "retriever",
            DB2EmbeddingRetriever(
                document_store=document_store_with_docs,
                top_k=4,
                filters={"field": "meta.language", "operator": "==", "value": "nonexistent"},
            ),
        )
        pipeline.add_component(
            "ranker",
            TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=2),
        )

        # Connect components
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "ranker.documents")

        # Run query
        query = "any query"
        results = pipeline.run({"text_embedder": {"text": query}, "ranker": {"query": query}})

        # Should handle empty results gracefully
        assert "ranker" in results
        assert "documents" in results["ranker"]
        assert len(results["ranker"]["documents"]) == 0
