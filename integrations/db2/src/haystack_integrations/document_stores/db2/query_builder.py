# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
QueryBuilder for DB2 SQL generation.

Provides a clean, maintainable way to generate SQL queries with proper
parameterization and extensibility.
"""

from typing import Any, Literal


class DB2QueryBuilder:
    """
    Builder for DB2 SQL queries with vector support.

    Centralizes SQL generation to improve maintainability, testability,
    and extensibility. Supports parameterized queries to prevent SQL injection.
    """

    def __init__(
        self,
        table_name: str,
        embedding_dimension: int,
        vector_type: str = "FLOAT32",
    ) -> None:
        """
        Initialize QueryBuilder.

        :param table_name: Name of the documents table.
        :param embedding_dimension: Dimension of embedding vectors.
        :param vector_type: DB2 vector type (default: FLOAT32).
        """
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.vector_type = vector_type

    def build_create_table(self) -> str:
        """
        Build CREATE TABLE statement.

        :return: SQL CREATE TABLE statement.
        """
        return f"CREATE TABLE {self.table_name} (id VARCHAR(255) NOT NULL PRIMARY KEY, content CLOB, embedding VECTOR({self.embedding_dimension}, {self.vector_type}), meta CLOB)"

    def build_drop_table(self) -> str:
        """
        Build DROP TABLE statement.

        :return: SQL DROP TABLE statement.
        """
        return f"DROP TABLE {self.table_name}"

    def build_table_exists(self) -> tuple[str, list[Any]]:
        """
        Build table existence check query.

        :return: Tuple of (SQL query, parameters).
        """
        # Handle schema-qualified table names (e.g., "schema.table")
        if "." in self.table_name:
            schema, table = self.table_name.split(".", 1)
            sql = "SELECT COUNT(*) as cnt FROM SYSCAT.TABLES WHERE TABNAME = ? AND TABSCHEMA = ?"
            params = [table.upper(), schema.upper()]
        else:
            sql = "SELECT COUNT(*) as cnt FROM SYSCAT.TABLES WHERE TABNAME = ? AND TABSCHEMA = CURRENT SCHEMA"
            params = [self.table_name.upper()]
        return sql, params

    def build_count_documents(self) -> str:
        """
        Build document count query.

        :return: SQL COUNT query.
        """
        return f"SELECT COUNT(*) as cnt FROM {self.table_name}"

    def build_document_exists(self) -> tuple[str, list[Any]]:
        """
        Build document existence check query.

        :param doc_id: Document ID to check.
        :return: Tuple of (SQL query, parameters).
        """
        sql = f"SELECT 1 FROM {self.table_name} WHERE id = ? FETCH FIRST 1 ROW ONLY"
        return sql, []  # Parameters will be bound separately

    def build_insert_document(self, embedding_str: str) -> tuple[str, list[Any]]:
        """
        Build INSERT document statement for single document.

        Note: embedding_str is embedded in SQL (not parameterized) because
        DB2 VECTOR type requires literal value in CAST expression.
        All user data (id, content, meta) is parameterized.

        :param embedding_str: String representation of embedding vector.
        :return: Tuple of (SQL query, parameter placeholders).
        """
        sql = f"""
        INSERT INTO {self.table_name} (id, content, embedding, meta)
        VALUES (?, ?, CAST('{embedding_str}' AS VECTOR({self.embedding_dimension}, {self.vector_type})), ?)
        """
        return sql, []  # Parameters (id, content, meta) will be bound separately

    def build_insert_documents_batch(self, documents_data: list[tuple[str, str, str, str]]) -> tuple[str, list[Any]]:
        """
        Build multi-row INSERT statement for batch insertion.

        Constructs a single INSERT with multiple VALUE clauses for improved performance.
        Each document's embedding is embedded in the SQL (DB2 VECTOR type requirement).

        Note: This method is more efficient than individual inserts but may hit SQL length
        limits with very large batches. Recommended batch size: 100-500 documents.

        :param documents_data: List of tuples (id, content, meta, embedding_str) for each document.
        :return: Tuple of (SQL query, parameter list for all documents).
        """
        if not documents_data:
            return "", []

        # Build VALUES clauses
        values_clauses = []
        params = []
        
        for doc_id, content, meta, embedding_str in documents_data:
            # Each VALUE clause: (?, ?, CAST('...' AS VECTOR), ?)
            values_clause = f"(?, ?, CAST('{embedding_str}' AS VECTOR({self.embedding_dimension}, {self.vector_type})), ?)"
            values_clauses.append(values_clause)
            # Add parameters for this document (id, content, meta)
            params.extend([doc_id, content, meta])

        # Combine into single INSERT statement
        values_sql = ",\n".join(values_clauses)
        sql = f"""
        INSERT INTO {self.table_name} (id, content, embedding, meta)
        VALUES {values_sql}
        """
        
        return sql, params

    def build_delete_by_ids(self, num_ids: int) -> tuple[str, list[Any]]:
        """
        Build DELETE by IDs statement.

        :param num_ids: Number of IDs to delete.
        :return: Tuple of (SQL query, parameter placeholders).
        """
        placeholders = ",".join(["?"] * num_ids)
        sql = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
        return sql, []  # Parameters will be bound separately

    def build_delete_by_filters(self, where_clause: str) -> str:
        """
        Build DELETE with filters statement.

        :param where_clause: WHERE clause from filter conversion.
        :return: SQL DELETE statement.
        """
        return f"DELETE FROM {self.table_name} WHERE {where_clause}"

    def build_select_by_id(self) -> tuple[str, list[Any]]:
        """
        Build SELECT by ID query.

        :return: Tuple of (SQL query, parameter placeholders).
        """
        sql = f"SELECT id, content, embedding, meta FROM {self.table_name} WHERE id = ?"
        return sql, []

    def build_select_by_filters(
        self,
        where_clause: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> str:
        """
        Build SELECT with filters query.

        :param where_clause: Optional WHERE clause from filter conversion.
        :param limit: Optional result limit for pagination.
        :param offset: Optional result offset for pagination.
        :return: SQL SELECT statement.
        """
        sql = f"SELECT id, content, embedding, meta FROM {self.table_name}"

        if where_clause:
            sql += f" WHERE {where_clause}"

        # DB2 pagination using OFFSET/FETCH
        if offset is not None and offset > 0:
            sql += f" OFFSET {offset} ROWS"

        if limit is not None:
            if offset is None or offset == 0:
                sql += f" FETCH FIRST {limit} ROWS ONLY"
            else:
                sql += f" FETCH NEXT {limit} ROWS ONLY"

        return sql

    def build_vector_search(
        self,
        distance_metric: Literal["cosine", "euclidean", "dot"],
        embedding_str: str,
        top_k: int,
        where_clause: str | None = None,
        offset: int = 0,
    ) -> str:
        """
        Build vector similarity search query.

        :param distance_metric: Distance metric to use.
        :param embedding_str: String representation of query embedding.
        :param top_k: Number of results to return.
        :param where_clause: Optional WHERE clause for filtering.
        :param offset: Number of results to skip (for pagination). Default is 0.
        :return: SQL SELECT statement with vector distance.
        """
        # Map distance metrics to DB2 VECTOR_DISTANCE function

        # Map distance metrics to DB2 metric types
        metric_map = {
            "cosine": "COSINE",
            "euclidean": "EUCLIDEAN",
            "dot": "DOT_PRODUCT",
        }
        db2_metric = metric_map.get(distance_metric, "COSINE")

        sql = f"""
        SELECT id, content, embedding, meta,
               VECTOR_DISTANCE(
                    embedding,
                    CAST('{embedding_str}' AS VECTOR({self.embedding_dimension}, {self.vector_type})),
                    {db2_metric}
                ) as distance
        FROM {self.table_name}
        """

        if where_clause:
            sql += f" WHERE {where_clause}"

        sql += " ORDER BY distance ASC"

        # Add pagination support using DB2 OFFSET/FETCH syntax
        if offset > 0:
            sql += f" OFFSET {offset} ROWS"
            sql += f" FETCH NEXT {top_k} ROWS ONLY"
        else:
            sql += f" FETCH FIRST {top_k} ROWS ONLY"

        return sql

    def build_keyword_search(
        self,
        query: str,
        top_k: int,
        where_clause: str | None = None,
        use_text_search: bool = False,
    ) -> str:
        """
        Build keyword search query with relevance scoring.

        Two modes available:
        1. Pattern matching (default): Uses LIKE for compatibility, scores by keyword frequency
        2. Text search (if use_text_search=True): Uses DB2 Text Search CONTAINS/SCORE for better relevance

        :param query: Search query string (can be multi-word).
        :param top_k: Number of results to return.
        :param where_clause: Optional WHERE clause for filtering.
        :param use_text_search: If True, use DB2 Text Search (requires text index). Default False for compatibility.
        :return: SQL SELECT statement with keyword matching and scoring.
        """
        if use_text_search:
            # Use DB2 Text Search with CONTAINS and SCORE functions
            # Requires: CREATE INDEX text_idx ON table_name(content) FOR TEXT
            escaped_query = query.replace("'", "''")

            sql = f"""
            SELECT id, content, embedding, meta,
                   SCORE({self.table_name}, content, '{escaped_query}') as score
            FROM {self.table_name}
            WHERE CONTAINS(content, '{escaped_query}') = 1
            """

            if where_clause:
                sql += f" AND {where_clause}"

            sql += f" ORDER BY score DESC FETCH FIRST {top_k} ROWS ONLY"

        else:
            # Fallback: Pattern matching with LIKE (works without text index)
            # Split query into keywords and escape special characters
            keywords = query.strip().split()
            escaped_keywords = [kw.replace("'", "''").replace("%", "\\%").replace("_", "\\_") for kw in keywords]

            # Build score calculation - count keyword frequency
            # More sophisticated scoring: count occurrences, not just presence
            score_cases = []
            for keyword in escaped_keywords:
                # Score based on number of occurrences of each keyword
                score_cases.append(
                    f"(LENGTH(LOWER(content)) - LENGTH(REPLACE(LOWER(content), LOWER('{keyword}'), ''))) / LENGTH('{keyword}')"
                )

            score_expr = " + ".join(score_cases) if score_cases else "0"

            sql = f"""
            SELECT id, content, embedding, meta,
                   ({score_expr}) as score
            FROM {self.table_name}
            """

            # Build WHERE clause - document must contain ALL keywords (AND logic)
            keyword_conditions = [f"LOWER(content) LIKE LOWER('%{keyword}%')" for keyword in escaped_keywords]

            conditions = []
            if keyword_conditions:
                conditions.append("(" + " AND ".join(keyword_conditions) + ")")
            if where_clause:
                conditions.append(where_clause)

            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

            # Sort by score DESC (higher scores = more keyword matches/frequency)
            sql += f" ORDER BY score DESC FETCH FIRST {top_k} ROWS ONLY"

        return sql

    def build_update_embedding(self, embedding_str: str) -> tuple[str, list[Any]]:
        """
        Build UPDATE embedding statement.

        :param embedding_str: String representation of new embedding.
        :return: Tuple of (SQL query, parameter placeholders).
        """
        sql = f"""
        UPDATE {self.table_name}
        SET embedding = CAST('{embedding_str}' AS VECTOR({self.embedding_dimension}, {self.vector_type}))
        WHERE id = ?
        """
        return sql, []

    def build_update_metadata(self) -> tuple[str, list[Any]]:
        """
        Build UPDATE metadata statement.

        :return: Tuple of (SQL query, parameter placeholders).
        """
        sql = f"UPDATE {self.table_name} SET meta = ? WHERE id = ?"
        return sql, []

    def build_update_document(self, embedding_str: str) -> tuple[str, list[Any]]:
        """
        Build UPDATE document statement (content, embedding, metadata).

        :param embedding_str: String representation of new embedding.
        :return: Tuple of (SQL query, parameter placeholders).
        """
        sql = f"""
        UPDATE {self.table_name}
        SET content = ?,
            embedding = CAST('{embedding_str}' AS VECTOR({self.embedding_dimension}, {self.vector_type})),
            meta = ?
        WHERE id = ?
        """
        return sql, []

    def build_count_with_filters(self, where_clause: str) -> str:
        """
        Build COUNT query with filters.

        :param where_clause: WHERE clause from filter conversion.
        :return: SQL COUNT statement.
        """
        return f"SELECT COUNT(*) as cnt FROM {self.table_name} WHERE {where_clause}"

    def build_select_all(self, include_embedding: bool = False) -> str:
        """
        Build SELECT all documents query.

        :param include_embedding: Whether to include embedding column.
        :return: SQL SELECT statement.
        """
        if include_embedding:
            return f"SELECT id, content, embedding, meta FROM {self.table_name}"
        return f"SELECT id, content, meta FROM {self.table_name}"
