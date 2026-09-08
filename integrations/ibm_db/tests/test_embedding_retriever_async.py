# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Async integration tests for IBM Db2 Embedding Retriever."""

from unittest.mock import AsyncMock, MagicMock

from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.ibm_db import IBMDb2EmbeddingRetriever
from haystack_integrations.document_stores.ibm_db import IBMDb2DocumentStore


class TestIBMDb2EmbeddingRetrieverAsync:
    """Unit tests for IBMDb2EmbeddingRetriever.run_async() using a mocked document store."""

    def _make_retriever(self, top_k=5, filters=None, filter_policy=FilterPolicy.REPLACE):
        """Build a retriever backed by a mock document store."""
        mock_store = MagicMock(spec=IBMDb2DocumentStore)
        doc = Document(id="doc1", content="result", embedding=[0.1, 0.2, 0.3, 0.4])
        mock_store._embedding_retrieval_async = AsyncMock(return_value=[doc])
        retriever = IBMDb2EmbeddingRetriever(
            document_store=mock_store,
            top_k=top_k,
            filters=filters or {},
            filter_policy=filter_policy,
        )
        return retriever, mock_store

    async def test_run_async_returns_documents(self):
        """run_async should delegate to _embedding_retrieval_async and return documents."""
        retriever, mock_store = self._make_retriever(top_k=5)
        result = await retriever.run_async(query_embedding=[0.1, 0.2, 0.3, 0.4])
        assert "documents" in result
        assert len(result["documents"]) == 1
        mock_store._embedding_retrieval_async.assert_awaited_once_with(
            [0.1, 0.2, 0.3, 0.4], filters={}, top_k=5
        )

    async def test_run_async_top_k_override(self):
        """Passing top_k to run_async overrides the constructor default."""
        retriever, mock_store = self._make_retriever(top_k=10)
        await retriever.run_async(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=3)
        assert mock_store._embedding_retrieval_async.call_args.kwargs["top_k"] == 3

    async def test_run_async_replace_policy_uses_runtime_filters(self):
        """REPLACE policy should use runtime filters, discarding constructor filters."""
        retriever, mock_store = self._make_retriever(
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
            filter_policy=FilterPolicy.REPLACE,
        )
        runtime_filters = {"field": "meta.year", "operator": ">", "value": 2020}
        await retriever.run_async(query_embedding=[0.1, 0.2, 0.3, 0.4], filters=runtime_filters)
        call_filters = mock_store._embedding_retrieval_async.call_args.kwargs["filters"]
        assert call_filters == runtime_filters

    async def test_run_async_merge_policy_combines_filters(self):
        """MERGE policy should combine constructor and runtime filters under AND."""
        retriever, mock_store = self._make_retriever(
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
            filter_policy=FilterPolicy.MERGE,
        )
        await retriever.run_async(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            filters={"field": "meta.year", "operator": ">", "value": 2020},
        )
        call_filters = mock_store._embedding_retrieval_async.call_args.kwargs["filters"]
        assert call_filters["operator"] == "AND"
        assert len(call_filters["conditions"]) == 2

    async def test_close_async_delegates_to_store(self):
        """close_async should call document_store.close_async()."""
        mock_store = MagicMock(spec=IBMDb2DocumentStore)
        mock_store.close_async = AsyncMock()
        retriever = IBMDb2EmbeddingRetriever(document_store=mock_store)
        await retriever.close_async()
        mock_store.close_async.assert_awaited_once()
