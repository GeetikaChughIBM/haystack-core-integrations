# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.ibm_db import IBMDb2EmbeddingRetriever
from haystack_integrations.document_stores.ibm_db import IBMDb2DocumentStore


@pytest.fixture
def store():
    return IBMDb2DocumentStore(
        database="testdb",
        hostname="localhost",
        username=MagicMock(resolve_value=lambda: "user"),
        password=MagicMock(resolve_value=lambda: "password"),
        embedding_dim=4,
    )


@pytest.mark.asyncio
async def test_run_async_returns_documents_and_applies_top_k(store):
    expected = [Document(id="doc", content="match", score=0.1)]
    store._embedding_retrieval_async = AsyncMock(return_value=expected)
    retriever = IBMDb2EmbeddingRetriever(document_store=store, top_k=5)

    result = await retriever.run_async([1.0, 0.0, 0.0, 0.0], top_k=2)

    assert result == {"documents": expected}
    store._embedding_retrieval_async.assert_awaited_once_with(
        [1.0, 0.0, 0.0, 0.0], filters={}, top_k=2
    )


@pytest.mark.asyncio
async def test_run_async_merges_constructor_filters(store):
    store._embedding_retrieval_async = AsyncMock(return_value=[])
    retriever = IBMDb2EmbeddingRetriever(
        document_store=store,
        filters={"field": "meta.lang", "operator": "==", "value": "python"},
        filter_policy=FilterPolicy.REPLACE,
    )

    await retriever.run_async([1.0, 0.0, 0.0, 0.0])

    store._embedding_retrieval_async.assert_awaited_once()
    assert store._embedding_retrieval_async.call_args.kwargs["filters"] == retriever.filters


@pytest.mark.asyncio
async def test_close_async_delegates_to_document_store(store):
    store.close_async = AsyncMock()
    await IBMDb2EmbeddingRetriever(document_store=store).close_async()
    store.close_async.assert_awaited_once_with()


def test_retriever_rejects_wrong_store():
    with pytest.raises(TypeError, match="IBMDb2DocumentStore"):
        IBMDb2EmbeddingRetriever(document_store=MagicMock())


def test_run_and_close_sync(store):
    store._embedding_retrieval = MagicMock(return_value=[])
    store.close = MagicMock()
    retriever = IBMDb2EmbeddingRetriever(document_store=store, top_k=3)
    assert retriever.run([1.0, 0.0, 0.0, 0.0]) == {"documents": []}
    retriever.close()
    store.close.assert_called_once_with()


def test_retriever_serialization(store):
    store.to_dict = MagicMock(return_value={"type": "store", "init_parameters": {}})
    retriever = IBMDb2EmbeddingRetriever(document_store=store, filters={"x": 1}, top_k=2)
    result = retriever.to_dict()
    assert result["init_parameters"]["top_k"] == 2
