# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Db2EmbeddingRetriever using live DB2 instance."""

import sys
from unittest.mock import Mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.ibm_db import Db2EmbeddingRetriever
from haystack_integrations.document_stores.ibm_db import Db2DocumentStore


@pytest.fixture
def document_store(connection_config, request):
    """Create a fresh document store for each test."""
    # Use test name to create unique table name per test
    table_name = f"test_retriever_emb_{request.node.name}_{sys.version_info.major}_{sys.version_info.minor}"

    store = Db2DocumentStore(
        connection_config=connection_config,
        table_name=table_name,
        embedding_dim=4,  # Small dimension for testing
        distance_metric="COSINE",
        recreate_table=True,
    )
    yield store
    # Cleanup after test
    try:
        conn = store._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {store.table_name}")
            conn.commit()
    except Exception:
        pass


@pytest.fixture
def sample_documents():
    """Create sample documents with embeddings for testing."""
    return [
        Document(
            id="doc1",
            content="Python programming language",
            meta={"category": "programming", "language": "python"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        ),
        Document(
            id="doc2",
            content="Java development",
            meta={"category": "programming", "language": "java"},
            embedding=[0.5, 0.6, 0.7, 0.8],
        ),
        Document(
            id="doc3",
            content="Data science with Python",
            meta={"category": "data-science", "language": "python"},
            embedding=[0.15, 0.25, 0.35, 0.45],
        ),
    ]


class TestDb2EmbeddingRetrieverUnit:
    """Unit tests for Db2EmbeddingRetriever that don't require a database."""

    def test_invalid_document_store_raises_type_error(self):
        """Test that invalid document store raises TypeError."""
        with pytest.raises(TypeError, match="must be an instance of Db2DocumentStore"):
            Db2EmbeddingRetriever(document_store="not_a_store")

    def test_init_with_filter_policy_string(self):
        """Test initialization with filter_policy as string."""
        mock_store = Mock(spec=Db2DocumentStore)

        retriever_replace = Db2EmbeddingRetriever(document_store=mock_store, filter_policy="replace")
        assert retriever_replace.filter_policy == FilterPolicy.REPLACE

        retriever_merge = Db2EmbeddingRetriever(document_store=mock_store, filter_policy="merge")
        assert retriever_merge.filter_policy == FilterPolicy.MERGE

    def test_init_with_filter_policy_enum(self):
        """Test initialization with FilterPolicy enum directly."""
        mock_store = Mock(spec=Db2DocumentStore)

        retriever_replace = Db2EmbeddingRetriever(document_store=mock_store, filter_policy=FilterPolicy.REPLACE)
        assert retriever_replace.filter_policy == FilterPolicy.REPLACE

        retriever_merge = Db2EmbeddingRetriever(document_store=mock_store, filter_policy=FilterPolicy.MERGE)
        assert retriever_merge.filter_policy == FilterPolicy.MERGE

    def test_to_dict(self, document_store):
        """Test serialization to dictionary."""
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            top_k=7,
            filters={"operator": "==", "field": "meta.x", "value": "y"},
        )
        d = retriever.to_dict()

        assert d["init_parameters"]["top_k"] == 7
        assert d["init_parameters"]["filters"] == {"operator": "==", "field": "meta.x", "value": "y"}
        assert d["init_parameters"]["filter_policy"] == "replace"
        assert "document_store" in d["init_parameters"]

    def test_from_dict(self, document_store):
        """Test deserialization from dictionary."""
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            top_k=7,
            filters={"operator": "==", "field": "meta.x", "value": "y"},
        )
        d = retriever.to_dict()

        restored = Db2EmbeddingRetriever.from_dict(d)
        assert restored.top_k == 7
        assert restored.filters == {"operator": "==", "field": "meta.x", "value": "y"}
        assert restored.filter_policy == FilterPolicy.REPLACE
        assert restored.document_store.table_name == document_store.table_name
        assert restored.document_store.embedding_dim == 4


@pytest.mark.integration
class TestDb2EmbeddingRetriever:
    """Test Db2EmbeddingRetriever with actual DB2 database."""

    def test_retriever_initialization(self, document_store):
        """Test retriever initialization."""
        retriever = Db2EmbeddingRetriever(document_store=document_store, top_k=5)
        assert retriever.document_store == document_store
        assert retriever.top_k == 5
        assert retriever.filters == {}
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_run_basic_retrieval(self, document_store, sample_documents):
        """Test basic embedding retrieval."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(document_store=document_store, top_k=2)

        # Query with embedding similar to doc1
        result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4])

        assert "documents" in result
        docs = result["documents"]
        assert len(docs) <= 2
        assert all(isinstance(doc, Document) for doc in docs)
        # First result should be doc1 (exact match)
        assert docs[0].id == "doc1"
        assert docs[0].score is not None

    def test_run_with_filters(self, document_store, sample_documents):
        """Test retrieval with metadata filters."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            top_k=5,
        )

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4])
        docs = result["documents"]

        # Should only return Python documents
        assert len(docs) == 2
        assert all(doc.meta.get("language") == "python" for doc in docs)

    def test_run_with_runtime_filters_replace_policy(self, document_store, sample_documents):
        """Test that runtime filters replace constructor filters with REPLACE policy."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            filter_policy=FilterPolicy.REPLACE,
            top_k=5,
        )

        # Runtime filter should replace constructor filter
        runtime_filters = {"operator": "==", "field": "meta.language", "value": "java"}
        result = retriever.run(query_embedding=[0.5, 0.6, 0.7, 0.8], filters=runtime_filters)
        docs = result["documents"]

        # Should only return Java documents (runtime filter)
        assert len(docs) == 1
        assert docs[0].meta.get("language") == "java"

    def test_run_with_runtime_filters_merge_policy(self, document_store, sample_documents):
        """Test that runtime filters merge with constructor filters with MERGE policy."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.category", "value": "programming"},
            filter_policy=FilterPolicy.MERGE,
            top_k=5,
        )

        # Runtime filter should merge with constructor filter
        runtime_filters = {"operator": "==", "field": "meta.language", "value": "python"}
        result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4], filters=runtime_filters)
        docs = result["documents"]

        # Should return only Python programming documents (both filters applied)
        assert len(docs) == 1
        assert docs[0].id == "doc1"
        assert docs[0].meta.get("category") == "programming"
        assert docs[0].meta.get("language") == "python"

    def test_run_top_k_override(self, document_store, sample_documents):
        """Test that runtime top_k overrides constructor top_k."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(document_store=document_store, top_k=10)

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=1)
        docs = result["documents"]

        assert len(docs) == 1

    def test_run_empty_store(self, document_store):
        """Test retrieval from empty store."""
        retriever = Db2EmbeddingRetriever(document_store=document_store, top_k=5)
        result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4])

        assert result["documents"] == []

    @pytest.mark.asyncio
    async def test_run_async(self, document_store, sample_documents):
        """Test async retrieval."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(document_store=document_store, top_k=2)

        result = await retriever.run_async(query_embedding=[0.1, 0.2, 0.3, 0.4])

        assert "documents" in result
        docs = result["documents"]
        assert len(docs) <= 2
        assert docs[0].id == "doc1"

    @pytest.mark.asyncio
    async def test_run_async_with_filters_and_top_k(self, document_store, sample_documents):
        """Test async retrieval with filters and custom top_k."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.category", "value": "programming"},
            top_k=10,
        )

        result = await retriever.run_async(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            top_k=1,
        )

        assert "documents" in result
        docs = result["documents"]
        assert len(docs) == 1
        assert docs[0].id == "doc1"
        assert docs[0].meta.get("language") == "python"

    @pytest.mark.asyncio
    async def test_run_async_matches_sync(self, document_store, sample_documents):
        """Test that async results match sync results."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            top_k=5,
        )

        query_embedding = [0.1, 0.2, 0.3, 0.4]

        # Get sync results
        sync_result = retriever.run(query_embedding=query_embedding)

        # Get async results
        async_result = await retriever.run_async(query_embedding=query_embedding)

        # Compare results
        assert len(sync_result["documents"]) == len(async_result["documents"])
        for sync_doc, async_doc in zip(sync_result["documents"], async_result["documents"], strict=True):
            assert sync_doc.id == async_doc.id
            assert sync_doc.content == async_doc.content
            assert sync_doc.meta == async_doc.meta
            # Scores might have minor floating point differences
            if sync_doc.score is not None and async_doc.score is not None:
                assert abs(sync_doc.score - async_doc.score) < 1e-6

    def test_serialization_round_trip(self, document_store):
        """Test complete serialization round-trip preserves all parameters."""
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            top_k=7,
            filter_policy=FilterPolicy.MERGE,
        )

        # Serialize
        serialized = retriever.to_dict()

        # Deserialize
        restored = Db2EmbeddingRetriever.from_dict(serialized)

        # Verify all parameters match
        assert restored.top_k == retriever.top_k
        assert restored.filters == retriever.filters
        assert restored.filter_policy == retriever.filter_policy
        assert restored.document_store.table_name == document_store.table_name
        assert restored.document_store.embedding_dim == document_store.embedding_dim
        assert restored.document_store.distance_metric == document_store.distance_metric


class TestEmbeddingRetrieverEdgeCases:
    """Edge-case coverage for Db2EmbeddingRetriever."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("filter_policy", "expected_policy"),
        [("replace", FilterPolicy.REPLACE), ("merge", FilterPolicy.MERGE)],
    )
    def test_init_converts_filter_policy_strings(self, document_store, filter_policy, expected_policy):
        """Test string filter_policy values are converted to FilterPolicy."""
        retriever = Db2EmbeddingRetriever(document_store=document_store, filter_policy=filter_policy)

        assert retriever.filter_policy == expected_policy

    @pytest.mark.integration
    def test_init_with_invalid_filter_policy_string_raises(self, document_store):
        """Test invalid filter_policy string raises an error."""
        with pytest.raises(ValueError):
            Db2EmbeddingRetriever(document_store=document_store, filter_policy="invalid")

    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("filters", "top_k", "filter_policy"),
        [
            (None, 10, FilterPolicy.REPLACE),
            ({}, 3, FilterPolicy.MERGE),
            ({"operator": "==", "field": "meta.language", "value": "python"}, 1, "replace"),
            ({"operator": "==", "field": "meta.category", "value": "programming"}, 7, "merge"),
        ],
    )
    def test_init_with_parameter_combinations(self, document_store, filters, top_k, filter_policy):
        """Test initialization with multiple parameter combinations."""
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters=filters,
            top_k=top_k,
            filter_policy=filter_policy,
        )

        assert retriever.document_store == document_store
        assert retriever.filters == (filters or {})
        assert retriever.top_k == top_k
        expected_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy
        assert retriever.filter_policy == expected_policy

    @pytest.mark.integration
    @pytest.mark.parametrize("filter_policy", [FilterPolicy.REPLACE, FilterPolicy.MERGE])
    def test_to_dict_round_trip_with_filter_policy_values(self, document_store, filter_policy):
        """Test serialization round-trip preserves filter_policy values."""
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            top_k=4,
            filter_policy=filter_policy,
        )

        serialized = retriever.to_dict()
        restored = Db2EmbeddingRetriever.from_dict(serialized)

        assert serialized["init_parameters"]["filter_policy"] == filter_policy.value
        assert restored.filter_policy == filter_policy
        assert restored.filters == retriever.filters
        assert restored.top_k == retriever.top_k
        assert restored.document_store.table_name == document_store.table_name

    @pytest.mark.integration
    def test_from_dict_old_pipeline_format_without_filter_policy(self, document_store):
        """Test deserialization compatibility with old pipeline format."""
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            top_k=6,
        )
        serialized = retriever.to_dict()
        serialized["init_parameters"].pop("filter_policy")

        restored = Db2EmbeddingRetriever.from_dict(serialized)

        assert restored.filter_policy == FilterPolicy.REPLACE
        assert restored.filters == retriever.filters
        assert restored.top_k == 6
        assert restored.document_store.table_name == document_store.table_name

    @pytest.mark.integration
    def test_from_dict_with_missing_optional_parameters(self, document_store):
        """Test deserialization when optional parameters are omitted."""
        serialized = {
            "type": "haystack_integrations.components.retrievers.ibm_db.embedding_retriever.Db2EmbeddingRetriever",
            "init_parameters": {
                "document_store": document_store.to_dict(),
            },
        }

        restored = Db2EmbeddingRetriever.from_dict(serialized)

        assert restored.filters == {}
        assert restored.top_k == 10
        assert restored.filter_policy == FilterPolicy.REPLACE
        assert restored.document_store.table_name == document_store.table_name

    @pytest.mark.integration
    def test_run_merge_policy_with_complex_filters(self, document_store, sample_documents):
        """Test MERGE policy combines constructor and runtime filters."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={
                "operator": "AND",
                "conditions": [
                    {"operator": "in", "field": "meta.category", "value": ["programming", "data-science"]},
                    {"operator": "==", "field": "meta.language", "value": "python"},
                ],
            },
            filter_policy=FilterPolicy.MERGE,
            top_k=5,
        )

        runtime_filters = {"operator": "==", "field": "meta.category", "value": "data-science"}
        result = retriever.run(query_embedding=[0.15, 0.25, 0.35, 0.45], filters=runtime_filters)

        assert [doc.id for doc in result["documents"]] == ["doc3"]

    @pytest.mark.integration
    def test_run_replace_policy_with_complex_runtime_filters(self, document_store, sample_documents):
        """Test REPLACE policy ignores constructor filters and uses runtime filters."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            filter_policy=FilterPolicy.REPLACE,
            top_k=5,
        )

        runtime_filters = {
            "operator": "AND",
            "conditions": [
                {"operator": "==", "field": "meta.category", "value": "programming"},
                {"operator": "==", "field": "meta.language", "value": "java"},
            ],
        }
        result = retriever.run(query_embedding=[0.5, 0.6, 0.7, 0.8], filters=runtime_filters)

        assert [doc.id for doc in result["documents"]] == ["doc2"]

    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("init_filters", "runtime_filters", "expected_ids"),
        [
            (None, None, ["doc1", "doc3", "doc2"]),
            ({"operator": "==", "field": "meta.language", "value": "python"}, None, ["doc1", "doc3"]),
            (None, {"operator": "==", "field": "meta.language", "value": "java"}, ["doc2"]),
        ],
    )
    def test_run_handles_none_filters_combinations(
        self, document_store, sample_documents, init_filters, runtime_filters, expected_ids
    ):
        """Test filter handling when constructor or runtime filters are None."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters=init_filters,
            filter_policy=FilterPolicy.MERGE,
            top_k=5,
        )

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4], filters=runtime_filters)

        assert [doc.id for doc in result["documents"]] == expected_ids

    @pytest.mark.integration
    def test_run_runtime_filters_override_init_filters_with_replace_policy(self, document_store, sample_documents):
        """Test runtime filters override init filters when using REPLACE policy."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.category", "value": "programming"},
            filter_policy="replace",
            top_k=5,
        )

        result = retriever.run(
            query_embedding=[0.15, 0.25, 0.35, 0.45],
            filters={"operator": "==", "field": "meta.category", "value": "data-science"},
        )

        assert [doc.id for doc in result["documents"]] == ["doc3"]

    @pytest.mark.integration
    def test_run_with_empty_query_embedding_returns_empty_results(self, document_store, sample_documents):
        """Test run with empty query embedding."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(document_store=document_store, top_k=5)

        result = retriever.run(query_embedding=[])

        assert result["documents"] == []

    @pytest.mark.integration
    def test_run_with_mismatched_embedding_dimensions_raises(self, document_store, sample_documents):
        """Test run with mismatched embedding dimensions raises an error."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(document_store=document_store, top_k=5)

        with pytest.raises(Exception):
            retriever.run(query_embedding=[0.1, 0.2])

    @pytest.mark.integration
    @pytest.mark.parametrize(("top_k", "expected_count"), [(0, 0), (1, 1), (2, 2), (10, 3)])
    def test_run_with_various_top_k_values(self, document_store, sample_documents, top_k, expected_count):
        """Test run with different top_k values."""
        document_store.write_documents(sample_documents)
        retriever = Db2EmbeddingRetriever(document_store=document_store, top_k=5)

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=top_k)

        assert len(result["documents"]) == expected_count

    def test_run_passes_merged_filters_to_document_store(self):
        """Test run passes merged filters to document store retrieval."""
        document_store = Mock(spec=Db2DocumentStore)
        document_store._embedding_retrieval.return_value = []

        retriever = Db2EmbeddingRetriever(
            document_store=document_store,
            filters={"operator": "==", "field": "meta.language", "value": "python"},
            filter_policy=FilterPolicy.MERGE,
            top_k=3,
        )

        retriever.run(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            filters={"operator": "==", "field": "meta.category", "value": "programming"},
        )

        document_store._embedding_retrieval.assert_called_once_with(
            [0.1, 0.2, 0.3, 0.4],
            filters={
                "operator": "AND",
                "conditions": [
                    {"operator": "==", "field": "meta.language", "value": "python"},
                    {"operator": "==", "field": "meta.category", "value": "programming"},
                ],
            },
            top_k=3,
        )


# Made with Bob
