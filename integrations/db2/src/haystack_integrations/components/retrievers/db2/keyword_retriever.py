# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Keyword-based retriever for DB2 document store.

This retriever uses DB2's text search capabilities to find documents
based on keyword matching, complementing the embedding-based retriever
for hybrid search scenarios.
"""

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.db2 import DB2DocumentStore


@component
class DB2KeywordRetriever:
    """
    Retrieve documents from DB2DocumentStore using keyword/text search.

    This retriever uses multi-word keyword matching with LIKE-based search
    to find documents. All keywords must be present in the document, and
    documents are scored based on the number of matching keywords.

    Note: This implementation uses LIKE '%keyword%' matching rather than
    DB2's full-text search (CONTAINS/SCORE) for broader compatibility.
    It's designed to work alongside DB2EmbeddingRetriever for hybrid search scenarios.

    Usage example:
    ```python
    from haystack_integrations.document_stores.db2 import DB2DocumentStore
    from haystack_integrations.components.retrievers.db2 import DB2KeywordRetriever
    from haystack.utils import Secret

    document_store = DB2DocumentStore(
        database="TESTDB",
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        embedding_dimension=384
    )

    retriever = DB2KeywordRetriever(document_store=document_store, top_k=10)

    results = retriever.run(query="Nike running shoes")
    documents = results["documents"]
    ```

    For hybrid search, combine with DB2EmbeddingRetriever:
    ```python
    from haystack import Pipeline
    from haystack.components.joiners import DocumentJoiner
    from haystack.components.embedders import SentenceTransformersTextEmbedder

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    pipeline.add_component("embedding_retriever", DB2EmbeddingRetriever(document_store=document_store))
    pipeline.add_component("keyword_retriever", DB2KeywordRetriever(document_store=document_store))
    pipeline.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion"))

    pipeline.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipeline.connect("embedding_retriever.documents", "joiner.documents")
    pipeline.connect("keyword_retriever.documents", "joiner.documents")
    ```
    """

    def __init__(
        self,
        *,
        document_store: DB2DocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Initialize DB2KeywordRetriever.

        :param document_store: Instance of DB2DocumentStore.
        :param filters: Filters applied to retrieved documents.
        :param top_k: Maximum number of documents to return.
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If document_store is not a DB2DocumentStore instance.
        """
        if not isinstance(document_store, DB2DocumentStore):
            msg = "document_store must be an instance of DB2DocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize component to dictionary.

        :return: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DB2KeywordRetriever":
        """
        Deserialize component from dictionary.

        :param data: Dictionary to deserialize from.
        :return: Deserialized component.
        """
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = DB2DocumentStore.from_dict(doc_store_params)

        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)

        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from DB2 using keyword search.

        :param query: Search query string.
        :param filters: Filters applied to retrieved documents. The way runtime filters
                       are applied depends on the filter_policy chosen at initialization.
        :param top_k: Maximum number of documents to return.
        :return: Dictionary with key "documents" containing list of retrieved documents.
        :raises ValueError: If query is empty or None.
        """
        if not query or not query.strip():
            msg = "query must be a non-empty string"
            raise ValueError(msg)

        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k

        docs = self.document_store.query_by_keyword(
            query=query,
            filters=filters,
            top_k=top_k,
        )

        return {"documents": docs}
