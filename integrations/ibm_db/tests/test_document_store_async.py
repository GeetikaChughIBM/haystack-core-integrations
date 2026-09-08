# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Async integration tests for IBM Db2 Document Store."""

import math

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store_async import (
    CountDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
    UpdateByFilterAsyncTest,
    WriteDocumentsAsyncTest,
)

from haystack_integrations.document_stores.ibm_db import IBMDb2DocumentStore


@pytest.mark.integration
class TestDocumentStoreAsync(
    CountDocumentsAsyncTest,
    WriteDocumentsAsyncTest,
    DeleteDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    FilterDocumentsAsyncTest,
    UpdateByFilterAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
):
    """
    Test IBMDb2DocumentStore async API surface using Haystack's async mixin tests.

    This class inherits from Haystack's async mixin test classes which provide
    standardized tests for async document store implementations.
    """

    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]):
        """
        Assert that two lists of Documents are equal, ignoring order.

        DB2 returns documents ordered by ID.  Sort both lists before comparing.
        """
        received_sorted = sorted(received, key=lambda d: d.id)
        expected_sorted = sorted(expected, key=lambda d: d.id)

        assert len(received_sorted) == len(expected_sorted), (
            f"Different number of documents: {len(received_sorted)} vs {len(expected_sorted)}"
        )

        for i, (rec, exp) in enumerate(zip(received_sorted, expected_sorted, strict=True)):
            assert rec.id == exp.id, f"Document {i}: IDs don't match: {rec.id} vs {exp.id}"
            assert rec.content == exp.content, f"Document {i} ({rec.id}): Content doesn't match"
            assert rec.meta == exp.meta, f"Document {i} ({rec.id}): Meta doesn't match"

            if rec.embedding is None and exp.embedding is None:
                continue
            elif rec.embedding is None or exp.embedding is None:
                msg = f"Document {i} ({rec.id}): One embedding is None, the other is not"
                raise AssertionError(msg)
            else:
                assert len(rec.embedding) == len(exp.embedding)
                for j, (r_val, e_val) in enumerate(zip(rec.embedding, exp.embedding, strict=True)):
                    if not math.isclose(r_val, e_val, rel_tol=1e-6, abs_tol=1e-9):
                        msg = f"Document {i} ({rec.id}): Embedding value {j} doesn't match: {r_val} vs {e_val}"
                        raise AssertionError(msg)

    async def test_write_documents_async(self, document_store: IBMDb2DocumentStore):
        """IBM Db2 default DuplicatePolicy.NONE raises DuplicateDocumentError on re-write."""
        doc = Document(content="test doc")
        assert await document_store.write_documents_async([doc]) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async([doc])

    async def test_write_documents_async_overwrite_policy(self, document_store: IBMDb2DocumentStore):
        """OVERWRITE policy replaces an existing document without raising."""
        doc = Document(id="ow1", content="original")
        await document_store.write_documents_async([doc])
        updated = Document(id="ow1", content="updated")
        written = await document_store.write_documents_async([updated], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1
        results = await document_store.filter_documents_async({"operator": "==", "field": "id", "value": "ow1"})
        assert len(results) == 1
        assert results[0].content == "updated"

    async def test_get_metadata_field_unique_values_distinct_types_async(
        self, document_store: IBMDb2DocumentStore
    ):
        """
        DB2 JSON_VALUE RETURNING VARCHAR returns all values as strings.
        Test each type separately to avoid mixed-type deduplication issues.
        """
        docs = [
            Document(id="int1", content="a", meta={"int_field": 1}),
            Document(id="int2", content="b", meta={"int_field": 2}),
            Document(id="str1", content="c", meta={"str_field": "hello"}),
            Document(id="str2", content="d", meta={"str_field": "world"}),
        ]
        await document_store.write_documents_async(docs)

        int_values, int_count = await document_store.get_metadata_field_unique_values_async("int_field")
        assert int_count == 2
        assert len(int_values) == 2

        str_values, str_count = await document_store.get_metadata_field_unique_values_async("str_field")
        assert str_count == 2
        assert len(str_values) == 2
