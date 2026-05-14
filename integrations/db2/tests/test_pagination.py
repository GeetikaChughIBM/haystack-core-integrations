# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document


class TestPagination:
    """Test pagination support in filter_documents."""

    @pytest.mark.integration
    def test_pagination_with_limit(self, document_store_local):
        """Test filtering with limit parameter."""
        # Create 10 documents
        docs = [Document(content=f"Document {i}", meta={"index": i}, embedding=[float(i)] * 384) for i in range(10)]
        document_store_local.write_documents(docs)

        # Get first 5 documents
        results = document_store_local.filter_documents(limit=5)
        assert len(results) == 5

    @pytest.mark.integration
    def test_pagination_with_offset(self, document_store_local):
        """Test filtering with offset parameter."""
        # Create 10 documents
        docs = [Document(content=f"Document {i}", meta={"index": i}, embedding=[float(i)] * 384) for i in range(10)]
        document_store_local.write_documents(docs)

        # Skip first 5 documents
        results = document_store_local.filter_documents(offset=5)
        assert len(results) == 5

    @pytest.mark.integration
    def test_pagination_with_offset_and_limit(self, document_store_local):
        """Test filtering with both offset and limit."""
        # Create 20 documents
        docs = [Document(content=f"Document {i}", meta={"index": i}, embedding=[float(i)] * 384) for i in range(20)]
        document_store_local.write_documents(docs)

        # Get documents 5-9 (skip 5, take 5)
        results = document_store_local.filter_documents(offset=5, limit=5)
        assert len(results) == 5

    @pytest.mark.integration
    def test_pagination_with_filters(self, document_store_local):
        """Test pagination combined with filters."""
        # Create documents with different categories
        docs = [
            Document(
                content=f"Doc {i}", meta={"category": "A" if i < 10 else "B", "index": i}, embedding=[float(i)] * 384
            )
            for i in range(20)
        ]
        document_store_local.write_documents(docs)

        # Filter category A with pagination
        results = document_store_local.filter_documents(filters={"category": "A"}, offset=2, limit=5)
        assert len(results) == 5
        # Verify all results are category A
        for doc in results:
            assert doc.meta["category"] == "A"

    @pytest.mark.integration
    def test_pagination_edge_cases(self, document_store_local):
        """Test pagination edge cases."""
        # Create 5 documents
        docs = [Document(content=f"Document {i}", meta={"index": i}, embedding=[float(i)] * 384) for i in range(5)]
        document_store_local.write_documents(docs)

        # Offset beyond available documents
        results = document_store_local.filter_documents(offset=10)
        assert len(results) == 0

        # Limit larger than available documents
        results = document_store_local.filter_documents(limit=100)
        assert len(results) == 5

        # Zero offset (should return all)
        results = document_store_local.filter_documents(offset=0)
        assert len(results) == 5

    @pytest.mark.integration
    def test_query_by_embedding_with_offset(self, document_store_local):
        """Test query_by_embedding with offset parameter for pagination."""
        # Create 10 documents with different embeddings
        docs = [
            Document(content=f"Document {i}", meta={"index": i}, embedding=[float(i) * 0.1] * 384) for i in range(10)
        ]
        document_store_local.write_documents(docs)

        # Query embedding similar to document 0
        query_embedding = [0.0] * 384

        # Get first page (top 3)
        results_page1 = document_store_local.query_by_embedding(query_embedding=query_embedding, top_k=3, offset=0)
        assert len(results_page1) == 3

        # Get second page (next 3)
        results_page2 = document_store_local.query_by_embedding(query_embedding=query_embedding, top_k=3, offset=3)
        assert len(results_page2) == 3

        # Verify pages don't overlap
        page1_ids = {doc.id for doc in results_page1}
        page2_ids = {doc.id for doc in results_page2}
        assert len(page1_ids.intersection(page2_ids)) == 0

    @pytest.mark.integration
    def test_query_by_embedding_offset_beyond_results(self, document_store_local):
        """Test query_by_embedding with offset beyond available results."""
        # Create 5 documents
        docs = [
            Document(content=f"Document {i}", meta={"index": i}, embedding=[float(i) * 0.1] * 384) for i in range(5)
        ]
        document_store_local.write_documents(docs)

        query_embedding = [0.0] * 384

        # Offset beyond available results
        results = document_store_local.query_by_embedding(query_embedding=query_embedding, top_k=5, offset=10)
        assert len(results) == 0

    @pytest.mark.integration
    def test_query_by_embedding_pagination_with_filters(self, document_store_local):
        """Test query_by_embedding pagination combined with filters."""
        # Create documents with different categories
        docs = [
            Document(
                content=f"Doc {i}",
                meta={"category": "A" if i < 10 else "B", "index": i},
                embedding=[float(i) * 0.1] * 384,
            )
            for i in range(20)
        ]
        document_store_local.write_documents(docs)

        query_embedding = [0.0] * 384

        # Get first page of category A
        results_page1 = document_store_local.query_by_embedding(
            query_embedding=query_embedding, filters={"category": "A"}, top_k=3, offset=0
        )
        assert len(results_page1) == 3
        assert all(doc.meta["category"] == "A" for doc in results_page1)

        # Get second page of category A
        results_page2 = document_store_local.query_by_embedding(
            query_embedding=query_embedding, filters={"category": "A"}, top_k=3, offset=3
        )
        assert len(results_page2) == 3
        assert all(doc.meta["category"] == "A" for doc in results_page2)
