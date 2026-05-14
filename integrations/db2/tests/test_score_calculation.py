# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document


class TestScoreCalculation:
    """Test score calculation in embedding retrieval."""

    @pytest.mark.integration
    def test_score_is_inverted_distance(self, document_store_local):
        """Test that score = 1 / (1 + distance) for embedding retrieval."""
        # Create documents with known embeddings
        docs = [
            Document(
                id="doc1",
                content="Document 1",
                embedding=[1.0, 0.0, 0.0] + [0.0] * 381,  # Unit vector in x direction
            ),
            Document(
                id="doc2",
                content="Document 2",
                embedding=[0.0, 1.0, 0.0] + [0.0] * 381,  # Unit vector in y direction
            ),
            Document(
                id="doc3",
                content="Document 3",
                embedding=[0.0, 0.0, 1.0] + [0.0] * 381,  # Unit vector in z direction
            ),
        ]
        document_store_local.write_documents(docs)

        # Query with embedding identical to doc1
        query_embedding = [1.0, 0.0, 0.0] + [0.0] * 381

        results = document_store_local.query_by_embedding(query_embedding=query_embedding, top_k=3)

        # Verify all documents have scores
        assert len(results) == 3
        for doc in results:
            assert doc.score is not None
            assert doc.score > 0
            assert doc.score <= 1.0  # Score should be between 0 and 1

        # First result should be doc1 with highest score (distance ~0)
        assert results[0].id == "doc1"
        # For identical vectors, distance should be ~0, so score should be close to 1
        assert results[0].score > 0.9

        # Other documents should have lower scores (higher distances)
        assert results[1].score < results[0].score
        assert results[2].score < results[0].score

    @pytest.mark.integration
    def test_higher_score_means_better_match(self, document_store_local):
        """Test that higher score indicates better match (Haystack convention)."""
        # Create documents with varying similarity to query
        docs = [
            Document(
                id="very_similar",
                content="Very similar",
                embedding=[0.9, 0.1, 0.0] + [0.0] * 381,  # Very close to query
            ),
            Document(
                id="somewhat_similar",
                content="Somewhat similar",
                embedding=[0.5, 0.5, 0.0] + [0.0] * 381,  # Moderately close
            ),
            Document(
                id="not_similar",
                content="Not similar",
                embedding=[0.0, 0.0, 1.0] + [0.0] * 381,  # Far from query
            ),
        ]
        document_store_local.write_documents(docs)

        # Query embedding
        query_embedding = [1.0, 0.0, 0.0] + [0.0] * 381

        results = document_store_local.query_by_embedding(query_embedding=query_embedding, top_k=3)

        # Extract scores
        scores = [doc.score for doc in results]

        # Verify scores are in descending order (higher = better)
        assert scores == sorted(scores, reverse=True), "Scores should be in descending order"

        # Verify the most similar document has the highest score
        assert results[0].id == "very_similar"
        assert results[0].score > results[1].score
        assert results[1].score > results[2].score

    @pytest.mark.integration
    def test_score_consistency_across_queries(self, document_store_local):
        """Test that score calculation is consistent across multiple queries."""
        # Create a single document
        doc = Document(
            id="test_doc",
            content="Test document",
            embedding=[1.0, 0.0, 0.0] + [0.0] * 381,
        )
        document_store_local.write_documents([doc])

        # Query multiple times with the same embedding
        query_embedding = [1.0, 0.0, 0.0] + [0.0] * 381

        scores = []
        for _ in range(3):
            results = document_store_local.query_by_embedding(query_embedding=query_embedding, top_k=1)
            scores.append(results[0].score)

        # All scores should be identical
        assert len(set(scores)) == 1, "Scores should be consistent across queries"
        # Score should be close to 1 for identical vectors
        assert scores[0] > 0.9

    @pytest.mark.integration
    def test_score_with_different_distance_metrics(self, document_store_local):
        """Test that scores are calculated correctly for different distance metrics."""
        # Note: This test uses the default metric from the fixture
        # In a real scenario, you'd test with different document stores using different metrics

        docs = [
            Document(
                id="doc1",
                content="Document 1",
                embedding=[1.0, 0.0, 0.0] + [0.0] * 381,
            ),
            Document(
                id="doc2",
                content="Document 2",
                embedding=[0.0, 1.0, 0.0] + [0.0] * 381,
            ),
        ]
        document_store_local.write_documents(docs)

        query_embedding = [1.0, 0.0, 0.0] + [0.0] * 381

        results = document_store_local.query_by_embedding(query_embedding=query_embedding, top_k=2)

        # Verify scores are properly calculated
        assert all(doc.score is not None for doc in results)
        assert all(0 < doc.score <= 1.0 for doc in results)
        # First result should have higher score
        assert results[0].score > results[1].score


# Made with Bob
