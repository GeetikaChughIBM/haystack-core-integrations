# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
DB2 Hybrid Retriever combining embedding and keyword search with RRF fusion.
"""

from dataclasses import replace
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.db2 import DB2DocumentStore

logger = logging.getLogger(__name__)


# Constants for weight validation
_WEIGHT_TOLERANCE_LOWER = 0.99  # Allow small floating point errors
_WEIGHT_TOLERANCE_UPPER = 1.01


@component
class DB2HybridRetriever:
    """
    Hybrid retriever combining embedding and keyword search for DB2.

    Uses Reciprocal Rank Fusion (RRF) to merge results from:
    - Vector similarity search (embedding-based)
    - Keyword search (LIKE-based scoring)

    This provides the best of both worlds: semantic understanding from embeddings
    and exact keyword matching for specific terms.

    This retriever delegates execution to DB2DocumentStore public APIs
    (query_by_embedding and query_by_keyword) for proper abstraction and
    maintainability.

    Usage example:
    ```python
    import os

    from haystack.utils import Secret
    from haystack_integrations.document_stores.db2 import DB2DocumentStore
    from haystack_integrations.components.retrievers.db2 import DB2HybridRetriever

    use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
    port = int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000"))

    document_store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        hostname=os.getenv("DB2_HOSTNAME"),
        port=port,
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        embedding_dimension=384,
        use_ssl=use_ssl,
    )

    retriever = DB2HybridRetriever(
        document_store=document_store,
        embedding_weight=0.7,  # 70% weight to embeddings
        keyword_weight=0.3,    # 30% weight to keywords
        top_k=10
    )

    # Use in pipeline
    results = retriever.run(
        query="machine learning",
        query_embedding=[0.1, 0.2, ...],  # from embedder
        top_k=5
    )
    ```
    """

    def __init__(
        self,
        *,
        document_store: DB2DocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
        embedding_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = 60,
        min_score: float | None = None,
    ) -> None:
        """
        Initialize DB2HybridRetriever.

        :param document_store: DB2DocumentStore instance.
        :param filters: Filters to apply to both retrievers.
        :param top_k: Maximum number of documents to return.
        :param filter_policy: Policy for handling filters at runtime.
        :param embedding_weight: Weight for embedding search results (0.0-1.0).
        :param keyword_weight: Weight for keyword search results (0.0-1.0).
        :param rrf_k: RRF constant (default: 60, standard value from literature).
        :param min_score: Minimum score threshold for returned documents (optional).
        :raises ValueError: If document_store is not a DB2DocumentStore instance, or if weights don't sum to 1.0 or are out of range.
        """
        if not isinstance(document_store, DB2DocumentStore):
            msg = "document_store must be an instance of DB2DocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = filter_policy
        self.embedding_weight = embedding_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.min_score = min_score

        # Validate weights
        if not (0.0 <= embedding_weight <= 1.0):
            msg = f"embedding_weight must be between 0.0 and 1.0, got {embedding_weight}"
            raise ValueError(msg)

        if not (0.0 <= keyword_weight <= 1.0):
            msg = f"keyword_weight must be between 0.0 and 1.0, got {keyword_weight}"
            raise ValueError(msg)

        total_weight = embedding_weight + keyword_weight
        if not (_WEIGHT_TOLERANCE_LOWER <= total_weight <= _WEIGHT_TOLERANCE_UPPER):
            msg = f"embedding_weight + keyword_weight must sum to 1.0, got {total_weight}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        :return: Serialized retriever.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            embedding_weight=self.embedding_weight,
            keyword_weight=self.keyword_weight,
            rrf_k=self.rrf_k,
            min_score=self.min_score,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DB2HybridRetriever":
        """
        Deserialize from dictionary.

        :param data: Serialized retriever.
        :return: DB2HybridRetriever instance.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" in init_params:
            init_params["document_store"] = DB2DocumentStore.from_dict(init_params["document_store"])
        if "filter_policy" in init_params:
            init_params["filter_policy"] = FilterPolicy(init_params["filter_policy"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Run hybrid retrieval combining embedding and keyword search.

        :param query: Query string for keyword search.
        :param query_embedding: Query embedding for vector search.
        :param filters: Optional filters to apply (overrides init filters based on filter_policy).
        :param top_k: Optional top_k override.
        :return: Dictionary with 'documents' key containing retrieved documents sorted by combined hybrid score (descending).
        :raises ValueError: If query is empty or query_embedding is invalid.
        """
        # Validate inputs
        if not query or not query.strip():
            msg = "query must be a non-empty string"
            raise ValueError(msg)

        if not query_embedding:
            msg = "query_embedding must be provided"
            raise ValueError(msg)

        # Determine filters based on policy
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)

        # Determine top_k
        final_top_k = top_k if top_k is not None else self.top_k

        # Retrieve more documents from each method for better fusion
        # Standard practice: retrieve 2-3x top_k from each source
        retrieval_k = final_top_k * 2

        # 1. Embedding-based retrieval
        embedding_docs = self._embedding_retrieval(
            query_embedding=query_embedding,
            top_k=retrieval_k,
            filters=filters,
        )

        # 2. Keyword-based retrieval
        keyword_docs = self._keyword_retrieval(
            query=query,
            top_k=retrieval_k,
            filters=filters,
        )

        # 3. Fuse results using RRF
        fused_docs = self._reciprocal_rank_fusion(
            embedding_docs=embedding_docs,
            keyword_docs=keyword_docs,
            top_k=final_top_k,
        )

        # 4. Apply score threshold filtering if specified
        if self.min_score is not None:
            fused_docs = [doc for doc in fused_docs if doc.score is not None and doc.score >= self.min_score]

        return {"documents": fused_docs}

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[Document]:
        """
        Perform embedding-based retrieval using DocumentStore's public API.

        Uses query_by_embedding() instead of private _embedding_retrieval()
        to follow Haystack best practices and maintain proper abstraction.

        :param query_embedding: Query embedding vector.
        :param top_k: Number of documents to retrieve.
        :param filters: Filters to apply.
        :return: Retrieved documents with scores.
        """
        try:
            docs = self.document_store.query_by_embedding(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                return_embedding=False,
            )
            return docs
        except Exception as e:
            logger.warning(f"Embedding retrieval failed: {e}. Continuing with keyword results only.")
            return []

    def _keyword_retrieval(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[Document]:
        """
        Perform keyword-based retrieval using DocumentStore's public API.

        Uses query_by_keyword() instead of private _keyword_retrieval()
        to follow Haystack best practices and maintain proper abstraction.

        :param query: Query string.
        :param top_k: Number of documents to retrieve.
        :param filters: Filters to apply.
        :return: Retrieved documents with scores.
        """
        try:
            docs = self.document_store.query_by_keyword(
                query=query,
                top_k=top_k,
                filters=filters,
            )
            return docs
        except Exception as e:
            logger.warning(f"Keyword retrieval failed: {e}. Continuing with embedding results only.")
            return []

    def _reciprocal_rank_fusion(
        self,
        embedding_docs: list[Document],
        keyword_docs: list[Document],
        top_k: int,
    ) -> list[Document]:
        """
        Fuse results using Reciprocal Rank Fusion (RRF).

        RRF formula: score(d) = Σ 1 / (k + rank(d))
        where k is a constant (typically 60) and rank is the position in the result list.

        Note: Documents appearing in both result sets receive higher scores due to
        the additive nature of RRF scoring.

        :param embedding_docs: Documents from embedding search.
        :param keyword_docs: Documents from keyword search.
        :param top_k: Number of documents to return.
        :return: Fused and ranked documents.
        """
        # Build score maps: doc_id -> RRF score
        embedding_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}

        # Calculate RRF scores for embedding results
        for rank, doc in enumerate(embedding_docs, start=1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            embedding_scores[doc.id] = rrf_score

        # Calculate RRF scores for keyword results
        for rank, doc in enumerate(keyword_docs, start=1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            keyword_scores[doc.id] = rrf_score

        # Combine all unique documents
        all_doc_ids = set(embedding_scores.keys()) | set(keyword_scores.keys())
        doc_map: dict[str, Document] = {}

        # Build document map (prefer embedding docs for content)
        for doc in embedding_docs:
            doc_map[doc.id] = doc
        for doc in keyword_docs:
            if doc.id not in doc_map:
                doc_map[doc.id] = doc

        # Calculate weighted combined scores
        combined_scores: list[tuple[str, float]] = []
        for doc_id in all_doc_ids:
            emb_score = embedding_scores.get(doc_id, 0.0)
            kw_score = keyword_scores.get(doc_id, 0.0)

            # Weighted combination
            combined_score = self.embedding_weight * emb_score + self.keyword_weight * kw_score

            combined_scores.append((doc_id, combined_score))

        # Sort by combined score (descending)
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # Build final result list with scores
        result_docs: list[Document] = []
        for doc_id, score in combined_scores[:top_k]:
            doc = doc_map[doc_id]
            # Add hybrid score to document metadata (use replace to avoid mutation warning)
            # Use underscore prefix to namespace and avoid overwriting user metadata
            meta = doc.meta.copy() if doc.meta else {}
            meta["_hybrid_score"] = score
            meta["_embedding_score"] = embedding_scores.get(doc_id, 0.0)
            meta["_keyword_score"] = keyword_scores.get(doc_id, 0.0)

            # Create new document with score using dataclasses.replace
            doc_with_score = replace(doc, score=score, meta=meta)
            result_docs.append(doc_with_score)

        return result_docs
