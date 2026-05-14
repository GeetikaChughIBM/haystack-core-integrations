# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Document

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore


@pytest.mark.integration
class TestDB2EmbeddingRetriever:
    """Integration tests for DB2EmbeddingRetriever."""

    def test_retriever_init(self, document_store):
        """Test retriever initialization."""
        retriever = DB2EmbeddingRetriever(
            document_store=document_store,
            top_k=5,
            return_embedding=True,
        )

        assert retriever.document_store == document_store
        assert retriever.top_k == 5
        assert retriever.return_embedding is True

    def test_retriever_to_dict(self, document_store):
        """Test retriever serialization."""
        retriever = DB2EmbeddingRetriever(
            document_store=document_store,
            top_k=10,
            filters={"category": "AI"},
        )

        data = retriever.to_dict()

        assert "type" in data
        assert "init_parameters" in data
        assert data["init_parameters"]["top_k"] == 10
        assert data["init_parameters"]["filters"] == {"category": "AI"}

    def test_retriever_from_dict(self, document_store):
        """Test retriever deserialization."""
        retriever = DB2EmbeddingRetriever(
            document_store=document_store,
            top_k=10,
        )

        data = retriever.to_dict()
        restored = DB2EmbeddingRetriever.from_dict(data)

        assert restored.top_k == 10
        assert restored.document_store.table_name == document_store.table_name

    def test_retriever_run_basic(self, document_store):
        """Test basic retrieval."""
        # Create and index documents
        docs = [
            Document(
                id="doc1",
                content="Machine learning is a subset of AI",
                embedding=[0.1, 0.2, 0.3] + [0.0] * (document_store.embedding_dimension - 3),
                meta={"category": "AI"},
            ),
            Document(
                id="doc2",
                content="Python is a programming language",
                embedding=[0.5, 0.6, 0.7] + [0.0] * (document_store.embedding_dimension - 3),
                meta={"category": "Programming"},
            ),
            Document(
                id="doc3",
                content="Deep learning uses neural networks",
                embedding=[0.15, 0.25, 0.35] + [0.0] * (document_store.embedding_dimension - 3),
                meta={"category": "AI"},
            ),
        ]

        document_store.write_documents(docs)

        # Create retriever
        retriever = DB2EmbeddingRetriever(document_store=document_store, top_k=2)

        # Query with embedding similar to doc1
        query_embedding = [0.1, 0.2, 0.3] + [0.0] * (document_store.embedding_dimension - 3)
        results = retriever.run(query_embedding=query_embedding)

        assert "documents" in results
        assert len(results["documents"]) <= 2
        assert all(isinstance(doc, Document) for doc in results["documents"])

        # First result should be most similar (doc1)
        assert results["documents"][0].id == "doc1"

    def test_retriever_run_with_filters(self, document_store):
        """Test retrieval with metadata filters."""
        docs = [
            Document(
                id="doc1",
                content="AI document 1",
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
                content="AI document 2",
                embedding=[0.15] * document_store.embedding_dimension,
                meta={"category": "AI", "priority": 3},
            ),
        ]

        document_store.write_documents(docs)

        retriever = DB2EmbeddingRetriever(
            document_store=document_store,
            filters={"category": "AI"},
            top_k=10,
        )

        query_embedding = [0.1] * document_store.embedding_dimension
        results = retriever.run(query_embedding=query_embedding)

        # Should only return AI documents
        assert len(results["documents"]) == 2
        assert all(doc.meta["category"] == "AI" for doc in results["documents"])

    def test_retriever_run_with_top_k(self, document_store):
        """Test retrieval with different top_k values."""
        # Use varied embeddings to avoid zero-magnitude vectors (division by zero in COSINE)
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Document {i}",
                embedding=[0.1 * i + j * 0.01 for j in range(document_store.embedding_dimension)],
            )
            for i in range(10)
        ]

        document_store.write_documents(docs)

        retriever = DB2EmbeddingRetriever(document_store=document_store, top_k=3)

        # Use varied query embedding to avoid zero-magnitude
        query_embedding = [0.1 + j * 0.01 for j in range(document_store.embedding_dimension)]
        results = retriever.run(query_embedding=query_embedding)

        assert len(results["documents"]) == 3

    def test_retriever_run_override_top_k(self, document_store):
        """Test that run() can override instance top_k."""
        # Use varied embeddings to avoid zero-magnitude vectors (division by zero in COSINE)
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Document {i}",
                embedding=[0.1 * i + j * 0.01 for j in range(document_store.embedding_dimension)],
            )
            for i in range(10)
        ]

        document_store.write_documents(docs)

        retriever = DB2EmbeddingRetriever(document_store=document_store, top_k=3)

        # Use varied query embedding to avoid zero-magnitude
        query_embedding = [0.1 + j * 0.01 for j in range(document_store.embedding_dimension)]
        results = retriever.run(query_embedding=query_embedding, top_k=5)

        assert len(results["documents"]) == 5

    def test_retriever_run_with_return_embedding(self, document_store):
        """Test retrieval with return_embedding=True."""
        docs = [
            Document(
                id="doc1",
                content="Test document",
                embedding=[0.1] * document_store.embedding_dimension,
            )
        ]

        document_store.write_documents(docs)

        retriever = DB2EmbeddingRetriever(
            document_store=document_store,
            return_embedding=True,
        )

        query_embedding = [0.1] * document_store.embedding_dimension
        results = retriever.run(query_embedding=query_embedding)

        assert len(results["documents"]) == 1
        # Note: return_embedding functionality depends on DB2 implementation
        # This test verifies the parameter is accepted

    def test_retriever_run_empty_query_embedding(self, document_store):
        """Test that empty query embedding raises error."""
        retriever = DB2EmbeddingRetriever(document_store=document_store)

        with pytest.raises(ValueError, match="query_embedding must be provided"):
            retriever.run(query_embedding=[])

    def test_retriever_run_wrong_dimension(self, document_store):
        """Test that wrong embedding dimension raises error."""
        retriever = DB2EmbeddingRetriever(document_store=document_store)

        wrong_embedding = [0.1] * 100  # Wrong dimension

        with pytest.raises(ValueError, match=r"dimension.*does not match"):
            retriever.run(query_embedding=wrong_embedding)

    def test_retriever_run_no_documents(self, document_store):
        """Test retrieval when no documents exist."""
        retriever = DB2EmbeddingRetriever(document_store=document_store)

        query_embedding = [0.1] * document_store.embedding_dimension
        results = retriever.run(query_embedding=query_embedding)

        assert "documents" in results
        assert len(results["documents"]) == 0

    def test_retriever_with_score(self, document_store):
        """Test that retrieved documents include similarity scores."""
        docs = [
            Document(
                id="doc1",
                content="Test document",
                embedding=[0.1] * document_store.embedding_dimension,
            ),
            Document(
                id="doc2",
                content="Another document",
                embedding=[0.5] * document_store.embedding_dimension,
            ),
        ]

        document_store.write_documents(docs)

        retriever = DB2EmbeddingRetriever(document_store=document_store, top_k=2)

        query_embedding = [0.1] * document_store.embedding_dimension
        results = retriever.run(query_embedding=query_embedding)

        # Check that scores are present in metadata
        for doc in results["documents"]:
            assert "score" in doc.meta
            assert isinstance(doc.meta["score"], float)

    def test_retriever_different_distance_metrics(self, document_store_local):
        """Test retrieval with different distance metrics."""
        # This test uses document_store_local to control distance metric
        for metric in ["cosine", "euclidean"]:
            # Create new store with specific metric
            store = DB2DocumentStore(
                database=document_store_local._database,
                username=document_store_local._username,
                password=document_store_local._password,
                table_name=f"test_metric_{metric}",
                embedding_dimension=384,
                distance_metric=metric,
                recreate_table=True,
            )

            docs = [
                Document(
                    id="doc1",
                    content="Test",
                    embedding=[0.1] * 384,
                )
            ]

            store.write_documents(docs)

            retriever = DB2EmbeddingRetriever(document_store=store)
            query_embedding = [0.1] * 384
            results = retriever.run(query_embedding=query_embedding)

            assert len(results["documents"]) == 1

            # Cleanup
            store._drop_table_if_exists()
