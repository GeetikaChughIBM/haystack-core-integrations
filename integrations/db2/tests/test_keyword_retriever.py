# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for DB2KeywordRetriever component."""

import pytest
from haystack import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.db2 import DB2KeywordRetriever
from haystack_integrations.document_stores.db2 import DB2DocumentStore

pytestmark = pytest.mark.integration


class TestDB2KeywordRetrieverInit:
    """Test DB2KeywordRetriever initialization."""

    def test_init_default(self, document_store_env):
        """Test initialization with default parameters."""
        retriever = DB2KeywordRetriever(document_store=document_store_env)

        assert retriever.document_store == document_store_env
        assert retriever.top_k == 10
        assert retriever.filters == {}
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_init_custom_parameters(self, document_store_env):
        """Test initialization with custom parameters."""
        retriever = DB2KeywordRetriever(
            document_store=document_store_env,
            top_k=5,
            filters={"category": "AI"},
            filter_policy=FilterPolicy.MERGE,
        )

        assert retriever.top_k == 5
        assert retriever.filters == {"category": "AI"}
        assert retriever.filter_policy == FilterPolicy.MERGE

    def test_init_invalid_document_store(self):
        """Test that initialization fails with invalid document store."""
        with pytest.raises(ValueError, match="must be an instance of DB2DocumentStore"):
            DB2KeywordRetriever(document_store="not a document store")


class TestDB2KeywordRetrieverSerialization:
    """Test DB2KeywordRetriever serialization."""

    def test_to_dict(self, document_store_env):
        """Test serialization to dictionary."""
        retriever = DB2KeywordRetriever(
            document_store=document_store_env,
            top_k=5,
            filters={"category": "AI"},
            filter_policy=FilterPolicy.MERGE,
        )

        data = retriever.to_dict()

        assert "type" in data
        assert "init_parameters" in data
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["filters"] == {"category": "AI"}
        assert data["init_parameters"]["filter_policy"] == "merge"

    def test_from_dict(self, document_store_env):
        """Test deserialization from dictionary."""
        retriever = DB2KeywordRetriever(
            document_store=document_store_env,
            top_k=5,
            filters={"category": "AI"},
        )

        data = retriever.to_dict()
        restored = DB2KeywordRetriever.from_dict(data)

        assert restored.top_k == 5
        assert restored.filters == {"category": "AI"}
        assert restored.document_store.table_name == document_store_env.table_name


class TestDB2KeywordRetrieverBasicSearch:
    """Test basic keyword search functionality."""

    def test_run_basic_keyword_search(self, document_store_env):
        """Test basic keyword search."""
        # Create and index documents
        docs = [
            Document(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence",
                embedding=[0.1] * document_store_env.embedding_dimension,
                meta={"category": "AI"},
            ),
            Document(
                id="doc2",
                content="Python is a programming language",
                embedding=[0.2] * document_store_env.embedding_dimension,
                meta={"category": "Programming"},
            ),
            Document(
                id="doc3",
                content="Deep learning uses neural networks for machine learning",
                embedding=[0.3] * document_store_env.embedding_dimension,
                meta={"category": "AI"},
            ),
        ]

        document_store_env.write_documents(docs)

        # Create retriever
        retriever = DB2KeywordRetriever(document_store=document_store_env, top_k=10)

        # Search for "machine learning"
        results = retriever.run(query="machine learning")

        assert "documents" in results
        assert len(results["documents"]) > 0
        assert all(isinstance(doc, Document) for doc in results["documents"])

        # Documents containing both "machine" and "learning" should be returned
        doc_ids = {doc.id for doc in results["documents"]}
        assert "doc1" in doc_ids or "doc3" in doc_ids

    def test_run_single_keyword(self, document_store_env):
        """Test search with single keyword."""
        docs = [
            Document(
                id="doc1",
                content="Python programming language",
                embedding=[0.1] * document_store_env.embedding_dimension,
            ),
            Document(
                id="doc2",
                content="Java programming language",
                embedding=[0.2] * document_store_env.embedding_dimension,
            ),
            Document(
                id="doc3",
                content="Machine learning with Python",
                embedding=[0.3] * document_store_env.embedding_dimension,
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(document_store=document_store_env)
        results = retriever.run(query="Python")

        assert len(results["documents"]) >= 2
        doc_ids = {doc.id for doc in results["documents"]}
        assert "doc1" in doc_ids
        assert "doc3" in doc_ids

    def test_run_multi_word_keyword(self, document_store_env):
        """Test search with multiple keywords."""
        docs = [
            Document(
                id="doc1",
                content="Nike running shoes for athletes",
                embedding=[0.1] * document_store_env.embedding_dimension,
            ),
            Document(
                id="doc2",
                content="Adidas running gear",
                embedding=[0.2] * document_store_env.embedding_dimension,
            ),
            Document(
                id="doc3",
                content="Nike basketball shoes",
                embedding=[0.3] * document_store_env.embedding_dimension,
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(document_store=document_store_env)
        results = retriever.run(query="Nike running")

        # Should return doc1 (has both Nike and running)
        assert len(results["documents"]) >= 1
        assert results["documents"][0].id == "doc1"

    def test_run_case_insensitive(self, document_store_env):
        """Test that keyword search is case-insensitive."""
        docs = [
            Document(
                id="doc1",
                content="Python Programming Language",
                embedding=[0.1] * document_store_env.embedding_dimension,
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(document_store=document_store_env)

        # Test different cases
        for query in ["python", "PYTHON", "Python"]:
            results = retriever.run(query=query)
            assert len(results["documents"]) == 1
            assert results["documents"][0].id == "doc1"


class TestDB2KeywordRetrieverFiltering:
    """Test keyword retriever with filters."""

    def test_run_with_filters(self, document_store_env):
        """Test keyword search with metadata filters."""
        docs = [
            Document(
                id="doc1",
                content="Python machine learning tutorial",
                embedding=[0.1] * document_store_env.embedding_dimension,
                meta={"category": "AI", "difficulty": "beginner"},
            ),
            Document(
                id="doc2",
                content="Python web development guide",
                embedding=[0.2] * document_store_env.embedding_dimension,
                meta={"category": "Web", "difficulty": "intermediate"},
            ),
            Document(
                id="doc3",
                content="Advanced Python machine learning",
                embedding=[0.3] * document_store_env.embedding_dimension,
                meta={"category": "AI", "difficulty": "advanced"},
            ),
        ]

        document_store_env.write_documents(docs)

        # Search with filter
        retriever = DB2KeywordRetriever(
            document_store=document_store_env,
            filters={"category": "AI"},
        )

        results = retriever.run(query="Python")

        # Should only return AI category documents
        assert len(results["documents"]) == 2
        assert all(doc.meta["category"] == "AI" for doc in results["documents"])

    def test_run_with_runtime_filters(self, document_store_env):
        """Test that runtime filters override instance filters."""
        docs = [
            Document(
                id="doc1",
                content="Python tutorial",
                embedding=[0.1] * document_store_env.embedding_dimension,
                meta={"category": "AI"},
            ),
            Document(
                id="doc2",
                content="Python guide",
                embedding=[0.2] * document_store_env.embedding_dimension,
                meta={"category": "Web"},
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(
            document_store=document_store_env,
            filters={"category": "AI"},
            filter_policy=FilterPolicy.REPLACE,
        )

        # Runtime filter should replace instance filter
        results = retriever.run(query="Python", filters={"category": "Web"})

        assert len(results["documents"]) == 1
        assert results["documents"][0].meta["category"] == "Web"

    def test_run_with_filter_policy_merge(self, document_store_env):
        """Test filter policy MERGE."""
        docs = [
            Document(
                id="doc1",
                content="Python tutorial",
                embedding=[0.1] * document_store_env.embedding_dimension,
                meta={"category": "AI", "level": "beginner"},
            ),
            Document(
                id="doc2",
                content="Python guide",
                embedding=[0.2] * document_store_env.embedding_dimension,
                meta={"category": "AI", "level": "advanced"},
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(
            document_store=document_store_env,
            filters={"category": "AI"},
            filter_policy=FilterPolicy.MERGE,
        )

        # Runtime filter should merge with instance filter
        results = retriever.run(query="Python", filters={"level": "beginner"})

        assert len(results["documents"]) == 1
        assert results["documents"][0].id == "doc1"


class TestDB2KeywordRetrieverTopK:
    """Test top_k parameter."""

    def test_run_with_top_k(self, document_store_env):
        """Test that top_k limits results."""
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Python programming tutorial {i}",
                embedding=[0.1 * i] * document_store_env.embedding_dimension,
            )
            for i in range(10)
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(document_store=document_store_env, top_k=3)
        results = retriever.run(query="Python")

        assert len(results["documents"]) == 3

    def test_run_override_top_k(self, document_store_env):
        """Test that runtime top_k overrides instance top_k."""
        docs = [
            Document(
                id=f"doc{i}",
                content=f"Python programming tutorial {i}",
                embedding=[0.1 * i] * document_store_env.embedding_dimension,
            )
            for i in range(10)
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(document_store=document_store_env, top_k=3)
        results = retriever.run(query="Python", top_k=5)

        assert len(results["documents"]) == 5


class TestDB2KeywordRetrieverScoring:
    """Test keyword scoring functionality."""

    def test_run_with_score(self, document_store_env):
        """Test that documents include scores."""
        docs = [
            Document(
                id="doc1",
                content="Python Python Python",  # More matches
                embedding=[0.1] * document_store_env.embedding_dimension,
            ),
            Document(
                id="doc2",
                content="Python programming",  # Fewer matches
                embedding=[0.2] * document_store_env.embedding_dimension,
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(document_store=document_store_env)
        results = retriever.run(query="Python")

        # Check that scores are present
        for doc in results["documents"]:
            assert doc.score is not None
            assert isinstance(doc.score, (int, float))

        # Document with more matches should have higher score
        if len(results["documents"]) >= 2:
            assert results["documents"][0].score >= results["documents"][1].score


class TestDB2KeywordRetrieverErrorHandling:
    """Test error handling."""

    def test_run_empty_query(self, document_store_env):
        """Test that empty query raises error."""
        retriever = DB2KeywordRetriever(document_store=document_store_env)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            retriever.run(query="")

    def test_run_none_query(self, document_store_env):
        """Test that None query raises error."""
        retriever = DB2KeywordRetriever(document_store=document_store_env)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            retriever.run(query=None)

    def test_run_whitespace_query(self, document_store_env):
        """Test that whitespace-only query raises error."""
        retriever = DB2KeywordRetriever(document_store=document_store_env)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            retriever.run(query="   ")

    def test_run_no_documents(self, document_store_env):
        """Test retrieval when no documents exist."""
        retriever = DB2KeywordRetriever(document_store=document_store_env)
        results = retriever.run(query="Python")

        assert "documents" in results
        assert len(results["documents"]) == 0

    def test_run_no_matching_documents(self, document_store_env):
        """Test retrieval when no documents match."""
        docs = [
            Document(
                id="doc1",
                content="Java programming",
                embedding=[0.1] * document_store_env.embedding_dimension,
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(document_store=document_store_env)
        results = retriever.run(query="Python")

        assert len(results["documents"]) == 0


class TestDB2KeywordRetrieverEdgeCases:
    """Test edge cases."""

    def test_run_special_characters_in_query(self, document_store_env):
        """Test query with special characters."""
        docs = [
            Document(
                id="doc1",
                content="C++ programming language",
                embedding=[0.1] * document_store_env.embedding_dimension,
            ),
            Document(
                id="doc2",
                content="C# programming language",
                embedding=[0.2] * document_store_env.embedding_dimension,
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(document_store=document_store_env)

        # Should handle special characters
        results = retriever.run(query="C++")
        assert len(results["documents"]) >= 1

    def test_run_unicode_query(self, document_store_env):
        """Test query with Unicode characters."""
        docs = [
            Document(
                id="doc1",
                content="Python 编程语言",
                embedding=[0.1] * document_store_env.embedding_dimension,
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(document_store=document_store_env)
        results = retriever.run(query="编程")

        assert len(results["documents"]) >= 1

    def test_run_very_long_query(self, document_store_env):
        """Test with very long query string."""
        docs = [
            Document(
                id="doc1",
                content="Python programming",
                embedding=[0.1] * document_store_env.embedding_dimension,
            ),
        ]

        document_store_env.write_documents(docs)

        retriever = DB2KeywordRetriever(document_store=document_store_env)
        long_query = "Python " * 100  # Very long query

        # Should handle long queries without error
        results = retriever.run(query=long_query)
        assert "documents" in results

# Made with Bob
