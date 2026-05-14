# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive integration tests for DB2 Haystack integration.

Tests the DB2DocumentStore and DB2EmbeddingRetriever implementation
with real embeddings and Haystack pipelines.
"""

import pytest
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore

pytestmark = pytest.mark.integration


class TestDB2DocumentStoreBasicOperations:
    """Test basic document store operations."""

    def test_write_and_count_documents(self, document_store_local):
        """Test writing documents and counting them."""
        documents = [
            Document(
                id="doc1",
                content="Artificial intelligence is transforming technology.",
                embedding=[0.1] * 384,
                meta={"category": "AI", "year": 2024},
            ),
            Document(
                id="doc2",
                content="Machine learning models require large datasets.",
                embedding=[0.2] * 384,
                meta={"category": "ML", "year": 2024},
            ),
            Document(
                id="doc3",
                content="Python is widely used in data science.",
                embedding=[0.3] * 384,
                meta={"category": "Programming", "year": 2023},
            ),
        ]

        written = document_store_local.write_documents(documents)
        assert written == 3

        count = document_store_local.count_documents()
        assert count == 3

    def test_filter_documents(self, document_store_local):
        """Test filtering documents by metadata."""
        documents = [
            Document(
                id="doc1",
                content="AI document",
                embedding=[0.1] * 384,
                meta={"category": "AI"},
            ),
            Document(
                id="doc2",
                content="ML document",
                embedding=[0.2] * 384,
                meta={"category": "ML"},
            ),
        ]

        document_store_local.write_documents(documents)

        filtered = document_store_local.filter_documents(filters={"category": "AI"})
        assert len(filtered) == 1
        assert filtered[0].id == "doc1"

    def test_duplicate_policy_skip(self, document_store_local):
        """Test duplicate policy SKIP."""
        duplicate_doc = Document(
            id="doc1",
            content="Original content",
            embedding=[0.5] * 384,
            meta={"category": "Test"},
        )
        document_store_local.write_documents([duplicate_doc])

        # Try to write duplicate with SKIP policy
        duplicate_doc2 = Document(
            id="doc1",
            content="This should be skipped",
            embedding=[0.6] * 384,
            meta={"category": "Test2"},
        )
        written = document_store_local.write_documents([duplicate_doc2], policy=DuplicatePolicy.SKIP)
        assert written == 0

        # Verify original content is preserved
        docs = document_store_local.filter_documents()
        assert len(docs) == 1
        assert docs[0].content == "Original content"

    def test_duplicate_policy_overwrite(self, document_store_local):
        """Test duplicate policy OVERWRITE."""
        doc = Document(
            id="doc2",
            content="Original content",
            embedding=[0.25] * 384,
            meta={"category": "ML", "year": 2024},
        )
        document_store_local.write_documents([doc])

        # Overwrite with new content
        overwrite_doc = Document(
            id="doc2",
            content="Machine learning has been updated.",
            embedding=[0.26] * 384,
            meta={"category": "ML", "year": 2025},
        )
        written = document_store_local.write_documents([overwrite_doc], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1

        # Verify overwrite
        filtered = document_store_local.filter_documents(filters={"year": 2025})
        assert len(filtered) == 1
        assert filtered[0].content == "Machine learning has been updated."

    def test_delete_documents(self, document_store_local):
        """Test deleting documents."""
        documents = [
            Document(id="doc1", content="Doc 1", embedding=[0.1] * 384),
            Document(id="doc2", content="Doc 2", embedding=[0.2] * 384),
            Document(id="doc3", content="Doc 3", embedding=[0.3] * 384),
        ]
        document_store_local.write_documents(documents)

        document_store_local.delete_documents(["doc3"])
        count_after = document_store_local.count_documents()
        assert count_after == 2


class TestDB2EmbeddingsAndRetrieval:
    """Test with real embeddings using sentence-transformers."""

    def test_real_embeddings_and_semantic_search(self, document_store_local):
        """Test with real embeddings and semantic retrieval."""
        # Sample documents about different topics
        texts = [
            "The Eiffel Tower is located in Paris, France.",
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "The Great Wall of China is one of the Seven Wonders.",
            "JavaScript is commonly used for web development.",
            "Deep learning uses neural networks with multiple layers.",
            "The Statue of Liberty is in New York City.",
            "Java is an object-oriented programming language.",
        ]

        documents = [Document(id=f"doc{i}", content=text, meta={"index": i}) for i, text in enumerate(texts)]

        # Generate embeddings
        embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        embedder.warm_up()

        result = embedder.run(documents)
        embedded_documents = result["documents"]

        # Write documents with embeddings
        written = document_store_local.write_documents(embedded_documents, policy=DuplicatePolicy.OVERWRITE)
        assert written == len(embedded_documents)

        # Initialize retriever
        retriever = DB2EmbeddingRetriever(
            document_store=document_store_local,
            top_k=3,
        )

        # Test semantic search
        test_queries = [
            "Tell me about programming languages",
            "What are famous landmarks?",
            "Explain artificial intelligence",
        ]

        text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        text_embedder.warm_up()

        for query in test_queries:
            # Generate query embedding
            query_result = text_embedder.run(query)
            query_embedding = query_result["embedding"]

            # Retrieve similar documents
            results = retriever.run(query_embedding=query_embedding, top_k=3)
            retrieved_docs = results["documents"]

            assert len(retrieved_docs) <= 3
            assert all(doc.score is not None for doc in retrieved_docs)


class TestDB2PipelineIntegration:
    """Test DB2 integration in Haystack pipelines."""

    def test_pipeline_integration(self, document_store_local):
        """Test DB2 integration in a complete Haystack pipeline."""
        # Create indexing pipeline
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component(
            "embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        )

        # Create query pipeline
        query_pipeline = Pipeline()
        query_pipeline.add_component(
            "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        )
        query_pipeline.add_component("retriever", DB2EmbeddingRetriever(document_store=document_store_local, top_k=3))
        query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

        # Prepare documents
        documents = [
            Document(content="Python is great for data science and machine learning."),
            Document(content="The Eiffel Tower is a famous landmark in Paris."),
            Document(content="Neural networks are the foundation of deep learning."),
            Document(content="JavaScript is essential for modern web development."),
            Document(content="The pyramids of Egypt are ancient wonders."),
        ]

        # Index documents
        indexing_result = indexing_pipeline.run({"embedder": {"documents": documents}})
        embedded_docs = indexing_result["embedder"]["documents"]
        document_store_local.write_documents(embedded_docs)

        # Query the pipeline
        query = "What programming languages are good for AI?"
        query_result = query_pipeline.run({"text_embedder": {"text": query}})
        retrieved_docs = query_result["retriever"]["documents"]

        assert len(retrieved_docs) <= 3
        assert all(isinstance(doc, Document) for doc in retrieved_docs)
        assert all(doc.score is not None for doc in retrieved_docs)


class TestDB2Serialization:
    """Test serialization and deserialization."""

    def test_document_store_serialization(self, document_store_local):
        """Test document store serialization and deserialization."""
        # Serialize
        store_dict = document_store_local.to_dict()
        assert "type" in store_dict
        assert "init_parameters" in store_dict

        # Deserialize
        restored_store = DB2DocumentStore.from_dict(store_dict)
        assert restored_store.table_name == document_store_local.table_name
        assert restored_store.embedding_dimension == document_store_local.embedding_dimension
        assert restored_store.distance_metric == document_store_local.distance_metric

    def test_retriever_serialization(self, document_store_local):
        """Test retriever serialization and deserialization."""
        retriever = DB2EmbeddingRetriever(
            document_store=document_store_local,
            top_k=5,
        )

        # Serialize
        retriever_dict = retriever.to_dict()
        assert "type" in retriever_dict
        assert "init_parameters" in retriever_dict

        # Deserialize
        restored_retriever = DB2EmbeddingRetriever.from_dict(retriever_dict)
        assert restored_retriever.top_k == retriever.top_k
