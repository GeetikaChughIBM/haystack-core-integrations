# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.db2 import DB2DocumentStore

logger = logging.getLogger(__name__)


@component
class DB2EmbeddingRetriever:
    """
    Retriever component for DB2DocumentStore using vector similarity search.

    Performs semantic search by comparing query embeddings with document embeddings
    stored in DB2 using the `VECTOR_DISTANCE` function.

    Usage example:
    ```python
    import os

    from haystack.utils import Secret
    from haystack_integrations.document_stores.db2 import DB2DocumentStore
    from haystack_integrations.components.retrievers.db2 import DB2EmbeddingRetriever

    use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
    port = int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000"))

    document_store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        hostname=os.getenv("DB2_HOSTNAME"),
        port=port,
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        use_ssl=use_ssl,
    )

    retriever = DB2EmbeddingRetriever(document_store=document_store, top_k=5)

    # In a pipeline, the retriever receives query_embedding from an embedder
    results = retriever.run(query_embedding=[0.1, 0.2, ...])
    documents = results["documents"]
    ```
    """

    def __init__(
        self,
        *,
        document_store: DB2DocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        return_embedding: bool = False,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Initialize DB2EmbeddingRetriever.

        :param document_store: Instance of DB2DocumentStore.
        :param filters: Optional filters to apply to the search.
        :param top_k: Maximum number of documents to return.
        :param return_embedding: Whether to return document embeddings.
        :param filter_policy: Policy to determine how filters are applied (REPLACE or MERGE).
        :raises ValueError: If document_store is not a DB2DocumentStore instance.
        """
        if not isinstance(document_store, DB2DocumentStore):
            msg = "document_store must be an instance of DB2DocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.return_embedding = return_embedding
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        :return: Dictionary representation.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            return_embedding=self.return_embedding,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DB2EmbeddingRetriever":
        """
        Deserialize from dictionary.

        :param data: Dictionary representation.
        :return: DB2EmbeddingRetriever instance.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" in init_params:
            init_params["document_store"] = DB2DocumentStore.from_dict(init_params["document_store"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        return_embedding: bool | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents similar to the query embedding.

        :param query_embedding: Query embedding vector.
        :param filters: Optional filters (overrides instance filters).
        :param top_k: Number of documents to return (overrides instance top_k).
        :param return_embedding: Whether to return embeddings (overrides instance setting).
        :return: Dictionary with "documents" key containing retrieved documents.
        """
        if not query_embedding:
            msg = "query_embedding must be provided"
            raise ValueError(msg)

        if len(query_embedding) != self.document_store.embedding_dimension:
            msg = (
                f"query_embedding dimension ({len(query_embedding)}) does not match "
                f"document store dimension ({self.document_store.embedding_dimension})"
            )
            raise ValueError(msg)

        # Use provided parameters or fall back to instance defaults
        final_filters = apply_filter_policy(self.filter_policy, self.filters, filters) or {}
        top_k = top_k if top_k is not None else self.top_k
        return_embedding = return_embedding if return_embedding is not None else self.return_embedding

        try:
            documents = self._retrieve_documents(
                query_embedding=query_embedding,
                filters=final_filters,
                top_k=top_k,
                return_embedding=return_embedding,
            )

            logger.info(f"Retrieved {len(documents)} documents")
            return {"documents": documents}

        except Exception as e:
            msg = f"Error retrieving documents: {e}"
            logger.error(msg)
            raise DocumentStoreError(msg) from e

    def _retrieve_documents(
        self,
        query_embedding: list[float],
        filters: dict[str, Any],
        top_k: int,
        return_embedding: bool,
    ) -> list[Document]:
        """
        Internal method to retrieve documents from DB2.

        Uses the DocumentStore's public API instead of direct SQL access.
        This follows Haystack best practices and maintains proper abstraction.

        :param query_embedding: Query embedding vector.
        :param filters: Filters to apply.
        :param top_k: Number of documents to return.
        :param return_embedding: Whether to return embeddings.
        :return: List of retrieved documents.
        :raises DocumentStoreError: If query execution fails.
        """
        # Use the DocumentStore's public API for embedding-based retrieval
        # This is the correct Haystack pattern - retriever orchestrates, store executes
        try:
            documents = self.document_store.query_by_embedding(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                return_embedding=return_embedding,
            )
            return documents
        except Exception as e:
            msg = f"Error retrieving documents: {e}"
            logger.error(msg)
            raise DocumentStoreError(msg) from e
