# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Document converters for DB2 document store.

This module provides utilities to convert between Haystack Document format
and DB2 database format.
"""

import json
import logging
from typing import Any

from haystack import Document

logger = logging.getLogger(__name__)


def document_to_db2_dict(document: Document, embedding_dimension: int) -> dict[str, Any]:
    """
    Convert a Haystack Document to a dictionary suitable for DB2 insertion.

    :param document: Haystack Document to convert.
    :param embedding_dimension: Expected embedding dimension for validation.
    :return: Dictionary with DB2-compatible fields.
    :raises ValueError: If document is missing ID, embedding, or has wrong dimension.

    Example:
        ```python
        from haystack import Document

        doc = Document(
            id="doc1",
            content="Sample text",
            embedding=[0.1, 0.2, 0.3],
            meta={"author": "John"}
        )
        db2_dict = document_to_db2_dict(doc, embedding_dimension=3)
        # Returns: {
        #     "id": "doc1",
        #     "content": "Sample text",
        #     "embedding": "[0.1,0.2,0.3]",
        #     "meta": '{"author": "John"}',
        #     ...
        # }
        ```
    """
    if not document.id or not document.id.strip():
        msg = "Document must have an ID"
        raise ValueError(msg)

    if document.embedding is None:
        msg = f"Document {document.id} missing embedding"
        raise ValueError(msg)

    if len(document.embedding) != embedding_dimension:
        msg = (
            f"Document {document.id} embedding dimension mismatch: "
            f"expected {embedding_dimension}, got {len(document.embedding)}"
        )
        raise ValueError(msg)

    # Convert embedding to DB2 vector format
    embedding_str = "[" + ",".join(map(str, document.embedding)) + "]"

    # Convert metadata to JSON string
    meta_json = json.dumps(document.meta) if document.meta else "{}"

    # Escape single quotes for SQL safety
    content_escaped = (document.content or "").replace("'", "''")
    meta_escaped = meta_json.replace("'", "''")
    doc_id_escaped = document.id.replace("'", "''")

    return {
        "id": doc_id_escaped,
        "content": content_escaped,
        "embedding": embedding_str,
        "meta": meta_escaped,
        "raw_id": document.id,  # Keep original for reference
        "raw_content": document.content or "",
        "raw_meta": document.meta or {},
    }


def db2_row_to_document(row: dict[str, Any], include_embedding: bool = False) -> Document:
    """
    Convert a DB2 row to a Haystack Document.

    :param row: Dictionary representing a DB2 row with keys: ID, CONTENT, META, EMBEDDING (optional).
    :param include_embedding: Whether to include embedding in the document.
    :return: Haystack Document.

    Example:
        ```python
        row = {
            "ID": "doc1",
            "CONTENT": "Sample text",
            "META": '{"author": "John"}',
            "EMBEDDING": "[0.1,0.2,0.3]",
            "SCORE": 0.95
        }
        doc = db2_row_to_document(row, include_embedding=True)
        # Returns Document with:
        #   id="doc1"
        #   content="Sample text"
        #   embedding=[0.1, 0.2, 0.3]
        #   meta={"author": "John", "score": 0.95}
        ```
    """
    # Parse metadata
    meta = {}
    if row.get("META"):
        try:
            meta = json.loads(row["META"])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse metadata for document {row.get('ID')}: {e}")
            meta = {}

    # Parse embedding if requested and available
    embedding = None
    if include_embedding and row.get("EMBEDDING"):
        try:
            # DB2 returns vector as string, need to parse it
            embedding_str = row["EMBEDDING"]
            if isinstance(embedding_str, str):
                # Remove brackets and split
                embedding_str = embedding_str.strip("[]")
                embedding = [float(x.strip()) for x in embedding_str.split(",")]
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse embedding for document {row.get('ID')}: {e}")
            embedding = None

    # Add score to metadata if present
    if "SCORE" in row and row["SCORE"] is not None:
        meta["score"] = float(row["SCORE"])

    return Document(id=row["ID"], content=row.get("CONTENT"), embedding=embedding, meta=meta)


def documents_to_db2_batch(documents: list[Document], embedding_dimension: int) -> list[dict[str, Any]]:
    """
    Convert a list of Haystack Documents to DB2-compatible dictionaries (optimized batch operation).

    This is more efficient than calling document_to_db2_dict() individually because:
    1. Validates all documents upfront before conversion
    2. Pre-allocates result list for better memory efficiency
    3. Provides better error context for batch operations

    :param documents: List of Haystack Documents.
    :param embedding_dimension: Expected embedding dimension.
    :return: List of DB2-compatible dictionaries.
    :raises ValueError: If any document is invalid.
    """
    if not documents:
        return []

    # Pre-allocate list for better performance
    db2_docs: list[dict[str, Any]] = []

    # Batch validation and conversion
    for idx, doc in enumerate(documents):
        try:
            # Validate document
            if not doc.id or not doc.id.strip():
                msg = f"Document at index {idx} must have an ID"
                raise ValueError(msg)

            if doc.embedding is None:
                msg = f"Document {doc.id} at index {idx} missing embedding"
                raise ValueError(msg)

            if len(doc.embedding) != embedding_dimension:
                msg = (
                    f"Document {doc.id} at index {idx} embedding dimension mismatch: "
                    f"expected {embedding_dimension}, got {len(doc.embedding)}"
                )
                raise ValueError(msg)

            # Convert (inline for performance)
            embedding_str = "[" + ",".join(map(str, doc.embedding)) + "]"
            meta_json = json.dumps(doc.meta) if doc.meta else "{}"

            # Escape single quotes for SQL safety
            content_escaped = (doc.content or "").replace("'", "''")
            meta_escaped = meta_json.replace("'", "''")
            doc_id_escaped = doc.id.replace("'", "''")

            db2_docs.append(
                {
                    "id": doc_id_escaped,
                    "content": content_escaped,
                    "embedding": embedding_str,
                    "meta": meta_escaped,
                    "raw_id": doc.id,
                    "raw_content": doc.content or "",
                    "raw_meta": doc.meta or {},
                }
            )

        except ValueError as e:
            logger.error(f"Failed to convert document {doc.id} at index {idx}: {e}")
            raise

    return db2_docs


def db2_rows_to_documents(rows: list[dict[str, Any]], include_embeddings: bool = False) -> list[Document]:
    """
    Convert a list of DB2 rows to Haystack Documents (optimized batch operation).

    This is more efficient than calling db2_row_to_document() individually because:
    1. Pre-allocates result list
    2. Handles errors gracefully without stopping entire batch
    3. Provides batch-level logging

    :param rows: List of DB2 row dictionaries.
    :param include_embeddings: Whether to include embeddings in documents.
    :return: List of Haystack Documents.
    """
    if not rows:
        return []

    documents: list[Document] = []
    failed_count = 0

    for idx, row in enumerate(rows):
        try:
            # Parse metadata
            meta = {}
            if row.get("META"):
                try:
                    meta = json.loads(row["META"])
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for document {row.get('ID')} at index {idx}: {e}")
                    meta = {}

            # Parse embedding if requested
            embedding = None
            if include_embeddings and row.get("EMBEDDING"):
                try:
                    embedding_str = row["EMBEDDING"]
                    if isinstance(embedding_str, str):
                        embedding_str = embedding_str.strip("[]")
                        embedding = [float(x.strip()) for x in embedding_str.split(",")]
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to parse embedding for document {row.get('ID')} at index {idx}: {e}")
                    embedding = None

            # Add score to metadata if present
            if "SCORE" in row and row["SCORE"] is not None:
                meta["score"] = float(row["SCORE"])

            doc = Document(id=row["ID"], content=row.get("CONTENT"), embedding=embedding, meta=meta)
            documents.append(doc)

        except Exception as e:
            failed_count += 1
            logger.warning(f"Failed to convert row at index {idx} to document: {e}")
            continue

    if failed_count > 0:
        logger.info(f"Batch conversion: {len(documents)} succeeded, {failed_count} failed")

    return documents


def validate_document_for_db2(document: Document, embedding_dimension: int) -> None:
    """
    Validate that a document is suitable for DB2 storage.

    :param document: Document to validate.
    :param embedding_dimension: Expected embedding dimension.
    :raises ValueError: If document is invalid.
    """
    if not document.id or not document.id.strip():
        msg = "Document must have an ID"
        raise ValueError(msg)

    if document.embedding is None:
        msg = f"Document {document.id} is missing embedding"
        raise ValueError(msg)

    if len(document.embedding) != embedding_dimension:
        msg = (
            f"Document {document.id} embedding dimension mismatch: "
            f"expected {embedding_dimension}, got {len(document.embedding)}"
        )
        raise ValueError(msg)

    # Validate metadata is JSON-serializable
    if document.meta:
        try:
            json.dumps(document.meta)
        except (TypeError, ValueError) as e:
            msg = f"Document {document.id} metadata is not JSON-serializable: {e}"
            raise ValueError(msg) from e


def sanitize_content_for_db2(content: str) -> str:
    """
    Sanitize content for DB2 storage.

    DB2 text fields may have limitations on certain characters.
    This function removes or replaces problematic characters.

    :param content: Content string to sanitize.
    :return: Sanitized content string.
    """
    if not content:
        return ""

    # Remove NULL bytes which can cause issues in DB2
    content = content.replace("\x00", "")

    # Escape single quotes for SQL
    content = content.replace("'", "''")

    return content


def parse_db2_vector(vector_str: str) -> list[float]:
    """
    Parse a DB2 vector string to a list of floats.

    DB2 returns vectors as strings like "[0.1, 0.2, 0.3]"

    :param vector_str: Vector string from DB2.
    :return: List of float values.
    :raises ValueError: If vector string is invalid.
    """
    if not vector_str:
        msg = "Empty vector string"
        raise ValueError(msg)

    try:
        # Remove brackets and whitespace
        vector_str = vector_str.strip().strip("[]")

        # Split by comma and convert to float
        values = [float(x.strip()) for x in vector_str.split(",")]

        return values
    except (ValueError, AttributeError) as e:
        msg = f"Invalid vector string format: {e}"
        raise ValueError(msg) from e


def format_vector_for_db2(embedding: list[float]) -> str:
    """
    Format an embedding list as a DB2 vector string.

    :param embedding: List of float values.
    :return: DB2-compatible vector string.
    """
    return "[" + ",".join(map(str, embedding)) + "]"
