# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.ibm_db import IBMDb2DocumentStore


@pytest.fixture
def store():
    return IBMDb2DocumentStore(
        database="testdb", hostname="localhost", username=Secret.from_token("user"),
        password=Secret.from_token("password"), table_name="test_documents", embedding_dim=4,
    )


@pytest.fixture
def mock_db_driver():
    with patch("haystack_integrations.document_stores.ibm_db.document_store.ibm_db_dbi") as driver:
        connection = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (0,)
        cursor.fetchall.return_value = []
        connection.cursor.return_value.__enter__.return_value = cursor
        driver.pconnect.return_value = connection
        yield driver, connection, cursor


@pytest.mark.asyncio
async def test_async_surface_delegates_to_sync_methods(store, monkeypatch):
    doc = Document(id="doc", content="content")
    methods = {
        "count_documents": 3, "filter_documents": [doc], "write_documents": 1,
        "delete_documents": None, "delete_all_documents": 2, "delete_by_filter": 1,
        "update_by_filter": 1, "count_documents_by_filter": 2,
        "count_unique_metadata_by_filter": {"x": 1}, "get_metadata_fields_info": {"x": {"type": "text"}},
        "get_metadata_field_min_max": {"min": 1, "max": 2},
        "get_metadata_field_unique_values": (["a"], 1), "_embedding_retrieval": [doc], "close": None,
    }
    calls = {}
    for name, result in methods.items():
        def record(*args, _name=name, _result=result, **kwargs):
            calls[_name] = (args, kwargs)
            return _result
        monkeypatch.setattr(store, name, record)

    assert await store.count_documents_async() == 3
    assert await store.filter_documents_async({"x": 1}) == [doc]
    assert await store.write_documents_async([doc]) == 1
    assert await store.delete_documents_async(["doc"]) is None
    assert await store.delete_all_documents_async(True) == 2
    assert await store.delete_by_filter_async({"x": 1}) == 1
    assert await store.update_by_filter_async({"x": 1}, {"y": 2}) == 1
    assert await store.count_documents_by_filter_async({"x": 1}) == 2
    assert await store.count_unique_metadata_by_filter_async({"x": 1}, ["x"]) == {"x": 1}
    assert await store.get_metadata_fields_info_async() == {"x": {"type": "text"}}
    assert await store.get_metadata_field_min_max_async("x") == {"min": 1, "max": 2}
    assert await store.get_metadata_field_unique_values_async("x", "a", 1, 5, {"y": 2}) == (["a"], 1)
    assert await store._embedding_retrieval_async([1.0], filters={"x": 1}, top_k=2) == [doc]
    assert await store.close_async() is None
    assert calls["write_documents"][0][0] == [doc]
    assert calls["write_documents"][0][1].value == "none"
    assert calls["_embedding_retrieval"][0] == ([1.0],)
    assert calls["_embedding_retrieval"][1] == {"filters": {"x": 1}, "top_k": 2}


@pytest.mark.asyncio
async def test_async_error_empty_and_large_inputs(store, monkeypatch):
    monkeypatch.setattr(store, "count_documents", MagicMock(side_effect=DocumentStoreError("DB2 query failed")))
    with pytest.raises(DocumentStoreError, match="DB2 query failed"):
        await store.count_documents_async()
    monkeypatch.setattr(store, "write_documents", MagicMock(return_value=0))
    delete = MagicMock(return_value=None)
    monkeypatch.setattr(store, "delete_documents", delete)
    assert await store.write_documents_async([]) == 0
    ids = [f"doc-{i}" for i in range(1000)]
    await store.delete_documents_async(ids)
    delete.assert_called_once_with(ids)


def test_connection_and_parameterized_filter(store, mock_db_driver):
    driver, connection, _ = mock_db_driver
    assert store.count_documents() == 0
    driver.pconnect.assert_called_once()
    assert connection.commit.called
    _, params = store._build_where_clause({"field": "meta.author", "operator": "==", "value": "x'); DROP TABLE users;--"})
    assert params == ["x'); DROP TABLE users;--"]


def test_connection_failure_and_schema_failure(store, mock_db_driver):
    driver, connection, _ = mock_db_driver
    driver.pconnect.side_effect = RuntimeError("connection refused")
    with pytest.raises(RuntimeError, match="connection refused") as error:
        store.count_documents()
    assert "password" not in str(error.value).lower()

    driver.pconnect.side_effect = None
    store.schema = "APP"
    connection.cursor.return_value.__enter__.return_value.execute.side_effect = RuntimeError("schema unavailable")
    with pytest.raises(RuntimeError, match="Failed to set schema APP"):
        store.count_documents()


def test_write_policies_and_empty_operations(store, mock_db_driver):
    _, connection, cursor = mock_db_driver
    doc = Document(id="a", content="c", meta={"x": 1}, embedding=[1.0, 0.0, 0.0, 0.0])
    cursor.rowcount = 1
    assert store.write_documents([]) == 0
    assert store.write_documents([doc], DuplicatePolicy.NONE) == 1
    assert store.write_documents([doc], DuplicatePolicy.SKIP) == 1
    assert store.write_documents([doc], DuplicatePolicy.OVERWRITE) == 1
    cursor.executemany.side_effect = Exception("duplicate SQL0803N")
    with pytest.raises(DuplicateDocumentError):
        store.write_documents([doc])
    assert store.delete_by_filter(None) == 0
    assert store.update_by_filter(None, {"x": 2}) == 0
    assert store.count_documents_by_filter(None) == 0
    assert store.count_unique_metadata_by_filter(None, None) == {}
    assert store.delete_all_documents() == 0
    assert connection.commit.called


def test_filter_delete_and_conversion(store, mock_db_driver):
    _, _, cursor = mock_db_driver
    cursor.fetchall.return_value = [("a", "content", '{"x": 1}', "[1, 0, 0, 0]")]
    docs = store.filter_documents({"field": "meta.x", "operator": "==", "value": 1})
    assert docs[0].id == "a" and docs[0].embedding == [1.0, 0.0, 0.0, 0.0]
    cursor.rowcount = 2
    assert store.delete_by_filter({"field": "meta.x", "operator": "==", "value": 1}) == 2
    store.delete_documents([])
    store.delete_documents(["a", "b"])


def test_validation_and_metadata_security(store):
    with pytest.raises(ValueError, match="cannot be empty"):
        store._validate_embedding([])
    with pytest.raises(TypeError, match="must be a list"):
        store._validate_embedding("bad")
    with pytest.raises(TypeError, match="numeric"):
        store._validate_embedding(["bad"])
    with pytest.raises(ValueError, match="Invalid metadata field name"):
        store.get_metadata_field_unique_values("author'; DROP TABLE users;--")


def test_metadata_queries_and_retrieval(store, mock_db_driver):
    _, _, cursor = mock_db_driver
    cursor.fetchall.return_value = [("bad-json",), ('{"kind": "text"}',)]
    assert store.get_metadata_fields_info()["kind"]["type"] == "text"
    cursor.fetchall.return_value = [("a",), ("b",)]
    assert store.count_unique_metadata_by_filter(metadata_fields=["kind"]) == {"kind": 2}
    cursor.fetchone.return_value = (1, 3)
    assert store.get_metadata_field_min_max("count") == {"min": 1, "max": 3}
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = [("\"a\"",)]
    assert store.get_metadata_field_unique_values("kind") == (["a"], 1)
    cursor.fetchall.return_value = [("a", "c", '{"x": 1}', "[1,0,0,0]", 0.25)]
    assert store._embedding_retrieval([1.0, 0.0, 0.0, 0.0], top_k=1)[0].score == 0.25


def test_embedding_error_and_close(store, mock_db_driver):
    _, connection, cursor = mock_db_driver
    cursor.execute.side_effect = RuntimeError("connection dropped")
    with pytest.raises(RuntimeError, match="connection dropped"):
        store._embedding_retrieval([1.0, 0.0, 0.0, 0.0])
    cursor.execute.side_effect = None
    store.count_documents()
    store.close()
    assert store._connection is None
    connection.close.assert_called_once()


def test_filter_translator_branches(store):
    from haystack.errors import FilterError
    cases = [
        {"operator": "AND", "conditions": [{"field": "id", "operator": "==", "value": "a"}]},
        {"operator": "OR", "conditions": [{"field": "id", "operator": "==", "value": "a"}]},
        {"operator": "NOT", "conditions": [{"field": "id", "operator": "==", "value": "a"}]},
        {"field": "meta.a", "operator": "in", "value": [1, 2]},
        {"field": "meta.a", "operator": "not in", "value": [1, 2]},
        {"field": "meta.a", "operator": "!=", "value": 1},
        {"field": "meta.a", "operator": "==", "value": None},
        {"field": "meta.a", "operator": ">", "value": 1},
    ]
    for case in cases:
        assert store._build_where_clause(case)[0]
    with pytest.raises(FilterError):
        store._build_where_clause({"field": "a", "operator": "in", "value": []})
    with pytest.raises(FilterError):
        store._build_where_clause({"field": "a", "operator": "wat", "value": 1})


def test_embedding_parser_and_field_types():
    from haystack_integrations.document_stores.ibm_db.document_store import _parse_embedding

    assert _parse_embedding(None) is None
    assert _parse_embedding([1, 2]) == [1.0, 2.0]
    assert _parse_embedding("[1, 2]") == [1.0, 2.0]
    assert _parse_embedding("bad") is None
    assert _parse_embedding(object()) is None
