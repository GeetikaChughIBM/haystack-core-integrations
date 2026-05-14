# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for DB2 document converters module.
"""

import json

import pytest
from haystack import Document

from haystack_integrations.document_stores.db2.converters import (
    db2_row_to_document,
    db2_rows_to_documents,
    document_to_db2_dict,
    documents_to_db2_batch,
    format_vector_for_db2,
    parse_db2_vector,
    sanitize_content_for_db2,
    validate_document_for_db2,
)


class TestDocumentToDb2Dict:
    """Tests for document_to_db2_dict function."""

    def test_basic_conversion(self):
        """Test basic document conversion."""
        doc = Document(id="test_1", content="Test content", embedding=[0.1, 0.2, 0.3], meta={"key": "value"})

        result = document_to_db2_dict(doc, embedding_dimension=3)

        assert result["id"] == "test_1"
        assert result["content"] == "Test content"
        assert result["embedding"] == "[0.1,0.2,0.3]"
        assert result["meta"] == '{"key": "value"}'
        assert result["raw_id"] == "test_1"

    def test_missing_embedding(self):
        """Test error when embedding is missing."""
        doc = Document(id="test_1", content="Test")

        with pytest.raises(ValueError, match="missing embedding"):
            document_to_db2_dict(doc, embedding_dimension=3)

    def test_wrong_embedding_dimension(self):
        """Test error when embedding dimension doesn't match."""
        doc = Document(id="test_1", content="Test", embedding=[0.1, 0.2])

        with pytest.raises(ValueError, match="dimension mismatch"):
            document_to_db2_dict(doc, embedding_dimension=3)

    def test_special_characters_escaping(self):
        """Test that single quotes are properly escaped."""
        doc = Document(id="test'1", content="Test's content", embedding=[0.1, 0.2, 0.3], meta={"key": "value's"})

        result = document_to_db2_dict(doc, embedding_dimension=3)

        assert result["id"] == "test''1"
        assert result["content"] == "Test''s content"
        assert "value''s" in result["meta"]

    def test_empty_metadata(self):
        """Test conversion with empty metadata."""
        doc = Document(id="test_1", content="Test", embedding=[0.1, 0.2, 0.3])

        result = document_to_db2_dict(doc, embedding_dimension=3)

        assert result["meta"] == "{}"

    def test_empty_content(self):
        """Test conversion with empty content."""
        doc = Document(id="test_1", content=None, embedding=[0.1, 0.2, 0.3])

        result = document_to_db2_dict(doc, embedding_dimension=3)

        assert result["content"] == ""


class TestDb2RowToDocument:
    """Tests for db2_row_to_document function."""

    def test_basic_conversion(self):
        """Test basic row to document conversion."""
        row = {"ID": "test_1", "CONTENT": "Test content", "META": '{"key": "value"}', "EMBEDDING": "[0.1,0.2,0.3]"}

        doc = db2_row_to_document(row, include_embedding=True)

        assert doc.id == "test_1"
        assert doc.content == "Test content"
        assert doc.meta == {"key": "value"}
        assert doc.embedding == [0.1, 0.2, 0.3]

    def test_without_embedding(self):
        """Test conversion without including embedding."""
        row = {"ID": "test_1", "CONTENT": "Test content", "META": '{"key": "value"}', "EMBEDDING": "[0.1,0.2,0.3]"}

        doc = db2_row_to_document(row, include_embedding=False)

        assert doc.embedding is None

    def test_with_score(self):
        """Test conversion with score in row."""
        row = {"ID": "test_1", "CONTENT": "Test content", "META": '{"key": "value"}', "SCORE": 0.95}

        doc = db2_row_to_document(row)

        assert doc.meta["score"] == 0.95

    def test_invalid_json_metadata(self):
        """Test handling of invalid JSON metadata."""
        row = {"ID": "test_1", "CONTENT": "Test content", "META": "invalid json"}

        doc = db2_row_to_document(row)

        assert doc.meta == {}

    def test_null_metadata(self):
        """Test handling of null metadata."""
        row = {"ID": "test_1", "CONTENT": "Test content", "META": None}

        doc = db2_row_to_document(row)

        assert doc.meta == {}


class TestBatchConversions:
    """Tests for batch conversion functions."""

    def test_documents_to_db2_batch(self):
        """Test batch document to DB2 conversion."""
        docs = [
            Document(id="1", content="Test 1", embedding=[0.1, 0.2]),
            Document(id="2", content="Test 2", embedding=[0.3, 0.4]),
        ]

        result = documents_to_db2_batch(docs, embedding_dimension=2)

        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_db2_rows_to_documents(self):
        """Test batch DB2 row to document conversion."""
        rows = [
            {"ID": "1", "CONTENT": "Test 1", "META": "{}"},
            {"ID": "2", "CONTENT": "Test 2", "META": "{}"},
        ]

        docs = db2_rows_to_documents(rows)

        assert len(docs) == 2
        assert docs[0].id == "1"
        assert docs[1].id == "2"

    def test_batch_with_invalid_document(self):
        """Test batch conversion with invalid document."""
        docs = [
            Document(id="1", content="Test 1", embedding=[0.1, 0.2]),
            Document(id="2", content="Test 2"),  # Missing embedding
        ]

        with pytest.raises(ValueError):
            documents_to_db2_batch(docs, embedding_dimension=2)


class TestValidateDocument:
    """Tests for validate_document_for_db2 function."""

    def test_valid_document(self):
        """Test validation of valid document."""
        doc = Document(id="test_1", content="Test", embedding=[0.1, 0.2, 0.3], meta={"key": "value"})

        # Should not raise
        validate_document_for_db2(doc, embedding_dimension=3)

    def test_missing_id(self):
        """Test validation fails for missing ID."""
        # Note: Haystack's Document class auto-generates UUIDs for empty/None IDs,
        # so we manually set an empty ID after creation to test the validation
        doc = Document(content="Test", embedding=[0.1, 0.2, 0.3])
        doc.id = ""  # Manually set empty ID to test validation (test-only mutation)

        with pytest.raises(ValueError, match="must have an ID"):
            validate_document_for_db2(doc, embedding_dimension=3)

    def test_missing_embedding(self):
        """Test validation fails for missing embedding."""
        doc = Document(id="test_1", content="Test")

        with pytest.raises(ValueError, match="missing embedding"):
            validate_document_for_db2(doc, embedding_dimension=3)

    def test_wrong_dimension(self):
        """Test validation fails for wrong embedding dimension."""
        doc = Document(id="test_1", content="Test", embedding=[0.1, 0.2])

        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_document_for_db2(doc, embedding_dimension=3)

    def test_non_serializable_metadata(self):
        """Test validation fails for non-JSON-serializable metadata."""
        doc = Document(
            id="test_1",
            content="Test",
            embedding=[0.1, 0.2, 0.3],
            meta={"key": object()},  # Not JSON serializable
        )

        with pytest.raises(ValueError, match="not JSON-serializable"):
            validate_document_for_db2(doc, embedding_dimension=3)


class TestSanitizeContent:
    """Tests for sanitize_content_for_db2 function."""

    def test_remove_null_bytes(self):
        """Test removal of NULL bytes."""
        content = "Test\x00content"
        result = sanitize_content_for_db2(content)
        assert result == "Testcontent"

    def test_escape_single_quotes(self):
        """Test escaping of single quotes."""
        content = "Test's content"
        result = sanitize_content_for_db2(content)
        assert result == "Test''s content"

    def test_empty_content(self):
        """Test handling of empty content."""
        assert sanitize_content_for_db2("") == ""
        assert sanitize_content_for_db2(None) == ""

    def test_combined_sanitization(self):
        """Test combined sanitization."""
        content = "Test's\x00content"
        result = sanitize_content_for_db2(content)
        assert result == "Test''scontent"


class TestVectorParsing:
    """Tests for vector parsing and formatting functions."""

    def test_parse_db2_vector(self):
        """Test parsing DB2 vector string."""
        vector_str = "[0.1, 0.2, 0.3]"
        result = parse_db2_vector(vector_str)
        assert result == [0.1, 0.2, 0.3]

    def test_parse_vector_without_spaces(self):
        """Test parsing vector without spaces."""
        vector_str = "[0.1,0.2,0.3]"
        result = parse_db2_vector(vector_str)
        assert result == [0.1, 0.2, 0.3]

    def test_parse_empty_vector(self):
        """Test parsing empty vector string."""
        with pytest.raises(ValueError):
            parse_db2_vector("")

    def test_parse_invalid_vector(self):
        """Test parsing invalid vector string."""
        with pytest.raises(ValueError):
            parse_db2_vector("invalid")

    def test_format_vector_for_db2(self):
        """Test formatting vector for DB2."""
        embedding = [0.1, 0.2, 0.3]
        result = format_vector_for_db2(embedding)
        assert result == "[0.1,0.2,0.3]"

    def test_format_empty_vector(self):
        """Test formatting empty vector."""
        result = format_vector_for_db2([])
        assert result == "[]"

    def test_round_trip_conversion(self):
        """Test round-trip vector conversion."""
        original = [0.1, 0.2, 0.3]
        formatted = format_vector_for_db2(original)
        parsed = parse_db2_vector(formatted)
        assert parsed == original


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_content(self):
        """Test handling of very long content."""
        long_content = "x" * 100000
        doc = Document(id="test_1", content=long_content, embedding=[0.1, 0.2, 0.3])

        result = document_to_db2_dict(doc, embedding_dimension=3)
        assert len(result["content"]) == 100000

    def test_unicode_content(self):
        """Test handling of Unicode content."""
        doc = Document(id="test_1", content="Test 中文 العربية 🎉", embedding=[0.1, 0.2, 0.3])

        result = document_to_db2_dict(doc, embedding_dimension=3)
        assert "中文" in result["content"]
        assert "العربية" in result["content"]
        assert "🎉" in result["content"]

    def test_large_embedding(self):
        """Test handling of large embedding vectors."""
        large_embedding = [0.1] * 1536  # OpenAI embedding size
        doc = Document(id="test_1", content="Test", embedding=large_embedding)

        result = document_to_db2_dict(doc, embedding_dimension=1536)
        assert len(result["embedding"]) > 1000

    def test_complex_metadata(self):
        """Test handling of complex nested metadata."""
        doc = Document(
            id="test_1",
            content="Test",
            embedding=[0.1, 0.2, 0.3],
            meta={"nested": {"key": "value", "list": [1, 2, 3]}, "number": 42, "boolean": True},
        )

        result = document_to_db2_dict(doc, embedding_dimension=3)
        meta = json.loads(result["meta"].replace("''", "'"))
        assert meta["nested"]["key"] == "value"
        assert meta["number"] == 42
        assert meta["boolean"] is True
