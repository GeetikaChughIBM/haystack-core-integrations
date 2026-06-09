# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
import threading
from dataclasses import replace
from typing import Any, Literal, cast

import ibm_db
from haystack import Document, default_from_dict, default_to_dict, logging
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.document_stores.db2.converters import db2_row_to_document, document_to_db2_dict
from haystack_integrations.document_stores.db2.filters import convert_filters
from haystack_integrations.document_stores.db2.query_builder import DB2QueryBuilder

logger = logging.getLogger(__name__)

# Constants for DB2 configuration
DEFAULT_PORT = 50000
DEFAULT_SSL_PORT = 50001
DEFAULT_HOST = "localhost"
PROTOCOL = "TCPIP"
VECTOR_TYPE = "FLOAT32"
MAX_VECTOR_DIMENSION = 16000  # DB2 VECTOR type maximum dimension limit


class DB2DocumentStore:
    """
    Document store using IBM DB2 with native vector support.

    Supports explicit TCP/IP connection flow using either a full connection string
    or connection details supplied through environment-backed parameters.

    **Note on Async Support:**
    Async operations are **not supported** due to limitations in the ibm_db library, which does not provide
    async/await capabilities. The ibm_db library is synchronous-only and does not support asyncio.
    All async methods (`count_documents_async`, `write_documents_async`, `delete_documents_async`,
    `filter_documents_async`) will raise `NotImplementedError` with a clear explanation.

    If you need async support, consider:
    - Using a thread pool executor to run synchronous operations in the background
    - Waiting for ibm_db to add async support in future releases

    **Important Notes:**
    - Embedding retrieval sets `doc.score = 1 - distance` for cosine distance (higher score = better match).
      For cosine distance (range 0-2): score 1.0 = identical, score 0.0 = orthogonal, score -1.0 = opposite.
    - All batch operations use `self.batch_size` for consistency with user configuration.
    - Pagination (offset parameter) is supported in `query_by_embedding()` for retrieving result pages.

    Usage example:
    ```python
    import os

    from haystack_integrations.document_stores.db2 import DB2DocumentStore
    from haystack.utils import Secret

    use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
    port = int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000"))

    store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        hostname=os.getenv("DB2_HOSTNAME"),
        port=port,
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        embedding_dimension=384,
        use_ssl=use_ssl,
        ssl_certificate=os.getenv("DB2_SSL_CERTIFICATE") or os.getenv("DB2_SSL_CERT_PATH"),
        batch_size=100,  # optional: control batch size for write operations
    )

    # Or use a full connection string from the environment
    store = DB2DocumentStore(
        connection_string=Secret.from_env_var("DB2_CONNECTION_STRING"),
        embedding_dimension=384
    )
    ```
    """

    def __init__(
        self,
        *,
        connection_string: Secret | None = None,
        database: str | None = None,
        username: Secret | None = None,
        password: Secret | None = None,
        hostname: str | None = None,
        port: int = DEFAULT_PORT,
        table_name: str = "haystack_documents",
        schema_name: str | None = None,
        embedding_dimension: int = 384,
        distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine",
        recreate_table: bool = False,
        batch_size: int = 100,
        embedding_model: str | None = None,
        validate_embedding_model: bool = True,
        use_ssl: bool = False,
        ssl_certificate: str | None = None,
        use_batch_insert: bool = True,
    ) -> None:
        """
        Initialize DB2DocumentStore.

        Connection methods (all use explicit TCP/IP):
        1. Connection string: provide `connection_string` with full connection details
        2. Parameter-based connection: provide `database`, `username`, `password`, `hostname`, and `port`

        In typical usage, these values come from environment variables such as
        `DB2_DATABASE`, `DB2_HOSTNAME`, `DB2_PORT`, `DB2_SSL_PORT`, `DB2_USER`,
        `DB2_PASSWORD`, and `DB2_SSL_ENABLED`.

        All connection methods use explicit TCP/IP protocol (`PROTOCOL=TCPIP`) for consistency.

        :param connection_string: Full DB2 connection string.
        :param database: Database name.
        :param username: Database username.
        :param password: Database password.
        :param hostname: Database hostname. 
        :param port: Database port. Use `DB2_PORT` for non-SSL connections and `DB2_SSL_PORT` for SSL connections.
        :param table_name: Table name for documents.
        :param embedding_dimension: Embedding vector dimension.
        :param distance_metric: Distance metric (cosine, euclidean, dot).
        :param recreate_table: Drop and recreate table if True.
        :param batch_size: Number of documents to process in each batch during write operations.
                          Default is 100. Larger batches may improve performance but use more memory.
        :param embedding_model: Name/identifier of the embedding model used to generate embeddings.
                               Example: "sentence-transformers/all-MiniLM-L6-v2"
                               This is stored in the database and validated on subsequent connections
                               to prevent using incompatible embedding models.
        :param validate_embedding_model: Whether to validate embedding model consistency.
                                        If True (default), raises an error if the provided embedding_model
                                        doesn't match the model stored in the database.
                                        Set to False only if you're certain about model compatibility.
        :param use_ssl: Enable SSL/TLS encryption for the database connection (default: False).
                       When True, the connection will use SSL/TLS to encrypt data in transit.
                       In environment-driven setups, this typically maps to `DB2_SSL_ENABLED`.
        :param ssl_certificate: Path to SSL certificate file for server verification (optional).
                               If provided, the certificate will be used to verify the server's identity.
                               Example: "/path/to/server-cert.pem"
        :param use_batch_insert: Use multi-row INSERT for batch operations (default: True).
                                When True, uses a single INSERT statement with multiple VALUE clauses for better performance.
                                When False, uses individual INSERT statements in a loop.
                                Set to False if you encounter SQL length limits with very large batches.
        :raises DocumentStoreError: If parameters are invalid or connection fails.
        :raises ValueError: If embedding model validation fails (model mismatch detected).
        """
        # Store batch size and insertion strategy for write operations
        self.batch_size = batch_size
        self.use_batch_insert = use_batch_insert

        # Store embedding model configuration
        self.embedding_model = embedding_model
        self.validate_embedding_model = validate_embedding_model

        # Validate and sanitize parameters
        self.table_name = self._sanitize_table_name(table_name)
        self._validate_embedding_dimension(embedding_dimension)
        self._validate_distance_metric(distance_metric)
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        self._table_initialized = False  # Track if table setup is complete

        # Store connection parameters as Secret objects for secure serialization
        self._connection_string_secret = connection_string
        self._database = database
        self._username = username
        self._password = password
        self._hostname = hostname
        self._port = port
        self._use_ssl = use_ssl
        self._ssl_certificate = ssl_certificate

        # Build actual connection string for use
        if connection_string:
            conn_str = connection_string.resolve_value()

            # Normalize connection string to uppercase for checking
            conn_str_upper = conn_str.upper() if conn_str else ""

            # Add SSL parameters if enabled and not already present
            if use_ssl and conn_str:
                if "SECURITY=SSL" not in conn_str_upper:
                    if not conn_str.endswith(";"):
                        conn_str += ";"
                    conn_str += "SECURITY=SSL;"

                if ssl_certificate and "SSLSERVERCERTIFICATE=" not in conn_str_upper:
                    if not conn_str.endswith(";"):
                        conn_str += ";"
                    conn_str += f"SSLServerCertificate={ssl_certificate};"

            self._connection_string = conn_str
        elif database and username and password:
            user = username.resolve_value()
            pwd = password.resolve_value()
            resolved_hostname = hostname or DEFAULT_HOST
            resolved_port = DEFAULT_SSL_PORT if use_ssl and port == DEFAULT_PORT else port

            conn_str = (
                f"DATABASE={database};HOSTNAME={resolved_hostname};PORT={resolved_port};"
                f"PROTOCOL={PROTOCOL};UID={user};PWD={pwd};"
            )

            if use_ssl:
                conn_str += "SECURITY=SSL;"
                if ssl_certificate:
                    conn_str += f"SSLServerCertificate={ssl_certificate};"

            self._connection_string = conn_str
        else:
            msg = (
                "Provide either: (1) connection_string, or "
                "(2) database + username + password with optional hostname + port"
            )
            raise DocumentStoreError(msg)

        # Store schema and table names
        self.schema_name = schema_name
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric

        # Build fully qualified table name
        self.qualified_table_name = f"{schema_name}.{table_name}" if schema_name else table_name

        # Thread-local storage for connections (prevents race conditions)
        self._local = threading.local()
        # Lock to serialize connection creation (ibm_db.connect is not thread-safe)
        self._connect_lock = threading.Lock()

        # Initialize QueryBuilder for SQL generation
        self.query_builder = DB2QueryBuilder(
            table_name=self.qualified_table_name,
            embedding_dimension=self.embedding_dimension,
            vector_type=VECTOR_TYPE,
        )

        # Initialize tables (documents + metadata)
        if recreate_table:
            self._drop_table_if_exists()
            self._drop_metadata_table_if_exists()
        self._create_table_if_not_exists()
        self._create_metadata_table_if_not_exists()

        # Validate embedding model consistency
        if self.validate_embedding_model and self.embedding_model:
            self._validate_model_consistency()

        self._table_initialized = True

    @staticmethod
    def _sanitize_table_name(table_name: str) -> str:
        """
        Sanitize table name to ensure it's safe for SQL.
        Replaces invalid characters with underscores.

        :param table_name: Table name to sanitize.
        :return: Sanitized table name.
        :raises DocumentStoreError: If table name is empty.
        """
        if not table_name:
            msg = "Table name cannot be empty"
            raise DocumentStoreError(msg)

        # Replace any non-alphanumeric/underscore characters with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)

        # Ensure it doesn't start with a number (DB2 requirement)
        if sanitized[0].isdigit():
            sanitized = f"t_{sanitized}"

        return sanitized

    @staticmethod
    def _validate_embedding_dimension(dimension: int) -> None:
        """
        Validate embedding dimension.

        :param dimension: Embedding dimension to validate.
        :raises DocumentStoreError: If dimension is invalid.
        """
        if dimension <= 0:
            msg = f"Embedding dimension must be positive, got {dimension}"
            raise DocumentStoreError(msg)

        if dimension > MAX_VECTOR_DIMENSION:  # DB2 VECTOR type limit
            msg = f"Embedding dimension {dimension} exceeds DB2 VECTOR limit of 16000"
            raise DocumentStoreError(msg)

    @staticmethod
    def _validate_distance_metric(metric: str) -> None:
        """
        Validate distance metric.

        :param metric: Distance metric to validate.
        :raises DocumentStoreError: If metric is invalid.
        """
        valid_metrics = ["cosine", "euclidean", "dot"]
        if metric not in valid_metrics:
            msg = f"Invalid distance metric '{metric}'. Must be one of: {', '.join(valid_metrics)}"
            raise DocumentStoreError(msg)

    def _get_connection(self) -> Any:
        """
        Get or create thread-local DB2 connection.

        Each thread gets its own connection to prevent race conditions.
        Connection creation is serialized with a lock because ibm_db.connect()
        is not thread-safe and crashes when called simultaneously.

        :return: Thread-local DB2 connection.
        :raises DocumentStoreError: If connection fails.
        """
        # Check if this thread already has a connection
        if not hasattr(self._local, "conn") or self._local.conn is None:
            # Serialize connection creation - ibm_db.connect() is not thread-safe
            with self._connect_lock:
                # Double-check after acquiring lock
                if not hasattr(self._local, "conn") or self._local.conn is None:
                    try:
                        self._local.conn = ibm_db.connect(self._connection_string, "", "")  # type: ignore[arg-type]
                        logger.debug(f"Created new DB2 connection for thread {threading.current_thread().name}")
                    except Exception as e:
                        msg = f"Failed to connect to DB2: {e}"
                        raise DocumentStoreError(msg) from e

        return self._local.conn

    def _table_exists(self) -> bool:
        """
        Check if the document table exists in the database.

        :return: True if table exists, False otherwise.
        """
        conn = self._get_connection()
        check_sql, params = self.query_builder.build_table_exists()

        try:
            stmt = ibm_db.prepare(conn, check_sql)
            if stmt is False:
                return False
            for idx, param in enumerate(params, 1):
                ibm_db.bind_param(stmt, idx, param)  # type: ignore[arg-type]
            ibm_db.execute(stmt)  # type: ignore[arg-type]
            row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
            if row is False or not isinstance(row, dict):
                return False
            return int(cast(Any, row["CNT"])) > 0
        except Exception as e:
            logger.debug(f"Table existence check failed: {e}")
            return False

    def _create_table_if_not_exists(self) -> None:
        """
        Create the document table if it doesn't already exist.

        :raises DocumentStoreError: If table creation fails.
        """
        if self._table_exists():
            logger.info(f"Table {self.table_name} exists")
            return

        conn = self._get_connection()
        create_sql = self.query_builder.build_create_table()

        try:
            ibm_db.exec_immediate(conn, create_sql)
            logger.info(f"Created table {self.qualified_table_name}")
        except Exception as e:
            msg = f"Failed to create table: {e}"
            raise DocumentStoreError(msg) from e

    def _drop_table_if_exists(self) -> None:
        """
        Drop the document table if it exists.

        Logs a warning if the drop operation fails but does not raise an exception.
        """
        if not self._table_exists():
            return

        conn = self._get_connection()
        try:
            drop_sql = self.query_builder.build_drop_table()
            ibm_db.exec_immediate(conn, drop_sql)
            logger.info(f"Dropped table {self.qualified_table_name}")
        except Exception as e:
            logger.warning(f"Error dropping table: {e}")

    def _get_metadata_table_name(self) -> str:
        """
        Get the fully qualified metadata table name.

        :return: Metadata table name, optionally schema-qualified.
        """
        metadata_table = f"{self.table_name}_metadata"
        return f"{self.schema_name}.{metadata_table}" if self.schema_name else metadata_table

    def _metadata_table_exists(self) -> bool:
        """
        Check if the metadata table exists in the database.

        :return: True if metadata table exists, False otherwise.
        """
        conn = self._get_connection()
        metadata_table_name = self._get_metadata_table_name()

        # Handle schema-qualified table names
        if "." in metadata_table_name:
            schema, table = metadata_table_name.split(".", 1)
            sql = "SELECT COUNT(*) as cnt FROM SYSCAT.TABLES WHERE TABNAME = ? AND TABSCHEMA = ?"
            stmt = ibm_db.prepare(conn, sql)
            ibm_db.bind_param(stmt, 1, table.upper())
            ibm_db.bind_param(stmt, 2, schema.upper())
        else:
            sql = "SELECT COUNT(*) as cnt FROM SYSCAT.TABLES WHERE TABNAME = ? AND TABSCHEMA = CURRENT SCHEMA"
            stmt = ibm_db.prepare(conn, sql)
            ibm_db.bind_param(stmt, 1, metadata_table_name.upper())

        ibm_db.execute(stmt)
        row = ibm_db.fetch_tuple(stmt)
        return row and int(row[0]) > 0

    def _create_metadata_table_if_not_exists(self) -> None:
        """Create metadata table to store embedding model information."""
        if self._metadata_table_exists():
            logger.info("Metadata table exists")
            return

        conn = self._get_connection()
        metadata_table_name = self._get_metadata_table_name()

        create_sql = f"""
        CREATE TABLE {metadata_table_name} (
            key VARCHAR(255) NOT NULL PRIMARY KEY,
            value VARCHAR(1000)
        )
        """

        try:
            ibm_db.exec_immediate(conn, create_sql)
            logger.info(f"Created metadata table {metadata_table_name}")
        except Exception as e:
            msg = f"Failed to create metadata table: {e}"
            raise DocumentStoreError(msg) from e

    def _drop_metadata_table_if_exists(self) -> None:
        """Drop metadata table if exists."""
        if not self._metadata_table_exists():
            return

        conn = self._get_connection()
        metadata_table_name = self._get_metadata_table_name()
        try:
            ibm_db.exec_immediate(conn, f"DROP TABLE {metadata_table_name}")
            logger.info(f"Dropped metadata table {metadata_table_name}")
        except Exception as e:
            logger.warning(f"Error dropping metadata table: {e}")

    def _get_metadata(self, key: str) -> str | None:
        """
        Get metadata value by key from the metadata table.

        :param key: Metadata key to retrieve.
        :return: Metadata value if found, None otherwise.
        """
        if not self._metadata_table_exists():
            return None

        conn = self._get_connection()
        metadata_table_name = self._get_metadata_table_name()

        sql = f"SELECT value FROM {metadata_table_name} WHERE key = ?"
        stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(stmt, 1, key)
        ibm_db.execute(stmt)

        row = ibm_db.fetch_tuple(stmt)
        return row[0] if row else None

    def _set_metadata(self, key: str, value: str) -> None:
        """
        Set metadata key-value pair in the metadata table.

        Uses UPDATE first, then INSERT if no rows were updated (upsert pattern).

        :param key: Metadata key to set.
        :param value: Metadata value to store.
        """
        conn = self._get_connection()
        metadata_table_name = self._get_metadata_table_name()

        # Try to update first
        update_sql = f"UPDATE {metadata_table_name} SET value = ? WHERE key = ?"
        stmt = ibm_db.prepare(conn, update_sql)
        ibm_db.bind_param(stmt, 1, value)
        ibm_db.bind_param(stmt, 2, key)
        ibm_db.execute(stmt)

        # If no rows updated, insert
        if ibm_db.num_rows(stmt) == 0:
            insert_sql = f"INSERT INTO {metadata_table_name} (key, value) VALUES (?, ?)"
            stmt = ibm_db.prepare(conn, insert_sql)
            ibm_db.bind_param(stmt, 1, key)
            ibm_db.bind_param(stmt, 2, value)
            ibm_db.execute(stmt)

    def _validate_model_consistency(self) -> None:
        """
        Validate that the embedding model is consistent with stored documents.

        Raises ValueError if model mismatch is detected.
        """
        stored_model = self._get_metadata("embedding_model")

        if stored_model:
            if stored_model != self.embedding_model:
                msg = (
                    f"\n{'=' * 80}\n"
                    f"EMBEDDING MODEL MISMATCH DETECTED!\n"
                    f"{'=' * 80}\n"
                    f"The documents in this database were indexed with a different embedding model.\n\n"
                    f"  Stored model:  {stored_model}\n"
                    f"  Current model: {self.embedding_model}\n\n"
                    f"Using different embedding models will produce MEANINGLESS search results!\n"
                    f"Even if both models have the same dimension, their embedding spaces are incompatible.\n\n"
                    f"SOLUTIONS:\n"
                    f"  1. Use the same model as stored: embedding_model='{stored_model}'\n"
                    f"  2. Re-index all documents with the new model: '{self.embedding_model}'\n"
                    f"  3. Set validate_embedding_model=False (NOT RECOMMENDED - only if you're certain)\n"
                    f"{'=' * 80}\n"
                )
                raise ValueError(msg)
            logger.info(f"Embedding model validated: {self.embedding_model}")
        # First time - store model info
        elif self.embedding_model:  # Type guard
            self._set_metadata("embedding_model", self.embedding_model)
            self._set_metadata("embedding_dimension", str(self.embedding_dimension))
            self._set_metadata("distance_metric", self.distance_metric)
            logger.info(f"Stored embedding model metadata: {self.embedding_model}")

    def count_documents(self, filters: dict[str, Any] | None = None) -> int:
        """
        Return number of documents, optionally filtered.

        :param filters: Optional filters to apply.
        :return: Document count.
        """
        conn = self._get_connection()
        params: list[Any] = []

        if filters:
            # Use the filter system from filters.py
            where_clause, params = convert_filters(filters)
            # Use QueryBuilder for consistent SQL generation
            sql = self.query_builder.build_count_with_filters(where_clause)
        else:
            # Use QueryBuilder for base count query
            sql = self.query_builder.build_count_documents()

        try:
            # Use proper parameter binding instead of string replacement
            if params:
                stmt = ibm_db.prepare(conn, sql)
                if not stmt:
                    msg = f"Failed to prepare statement: {ibm_db.conn_errormsg(conn)}"
                    raise DocumentStoreError(msg)

                for i, param in enumerate(params, 1):
                    ibm_db.bind_param(stmt, i, param)  # type: ignore[arg-type]

                ibm_db.execute(stmt)  # type: ignore[arg-type]
            else:
                stmt = ibm_db.exec_immediate(conn, sql)

            row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
            if row is False or not isinstance(row, dict):
                return 0
            return int(cast(Any, row["CNT"]))
        except Exception as e:
            msg = f"Error counting documents: {e}"
            raise DocumentStoreError(msg) from e

    def _document_exists(self, conn: Any, doc_id: str) -> bool:
        """
        Check if a document exists in the store.

        :param conn: Database connection.
        :param doc_id: Document ID to check.
        :return: True if document exists, False otherwise.
        """
        # Use QueryBuilder for consistent SQL generation
        check_sql, _ = self.query_builder.build_document_exists()
        stmt = ibm_db.prepare(conn, check_sql)
        if stmt is False:
            return False
        ibm_db.bind_param(stmt, 1, doc_id)  # type: ignore[arg-type]
        ibm_db.execute(stmt)  # type: ignore[arg-type]
        row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
        return row is not False and isinstance(row, dict)

    def _documents_exist_batch(self, conn: Any, doc_ids: list[str]) -> set[str]:
        """
        Check which documents exist in the store (batch operation).

        More efficient than calling _document_exists() for each document.

        :param conn: Database connection.
        :param doc_ids: List of document IDs to check.
        :return: Set of document IDs that exist in the store.
        """
        if not doc_ids:
            return set()

        # Build parameterized query with IN clause
        placeholders = ",".join(["?"] * len(doc_ids))
        check_sql = f"SELECT id FROM {self.qualified_table_name} WHERE id IN ({placeholders})"
        stmt = ibm_db.prepare(conn, check_sql)
        if stmt is False:
            return set()

        # Bind parameters
        for idx, doc_id in enumerate(doc_ids, 1):
            ibm_db.bind_param(stmt, idx, doc_id)  # type: ignore[arg-type]

        ibm_db.execute(stmt)  # type: ignore[arg-type]

        # Collect existing IDs
        existing_ids: set[str] = set()
        while True:
            row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
            if row is False or not isinstance(row, dict):
                break
            existing_ids.add(cast(str, row["ID"]))

        return existing_ids

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Write documents to store with configurable batch insertion strategy.

        Uses multi-row INSERT (use_batch_insert=True) or individual INSERTs (use_batch_insert=False).

        :param documents: Documents to write.
        :param policy: Duplicate handling policy.
        :return: Number written.
        """
        if not documents:
            return 0

        conn = self._get_connection()
        written = 0

        # Process in batches for better performance
        batch_size = self.batch_size if len(documents) >= self.batch_size else len(documents)

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Handle duplicate policy for batch
            if policy == DuplicatePolicy.SKIP:
                # Filter out existing documents using batch check
                batch_ids = [doc.id for doc in batch]
                existing_ids = self._documents_exist_batch(conn, batch_ids)
                batch = [doc for doc in batch if doc.id not in existing_ids]
            elif policy == DuplicatePolicy.OVERWRITE:
                # Delete existing documents in batch
                ids_to_delete = [doc.id for doc in batch]
                if ids_to_delete:
                    placeholders = ",".join(["?"] * len(ids_to_delete))
                    delete_sql = f"DELETE FROM {self.qualified_table_name} WHERE id IN ({placeholders})"
                    stmt = ibm_db.prepare(conn, delete_sql)
                    if stmt is False:
                        msg = "Failed to prepare delete statement"
                        raise DocumentStoreError(msg)
                    for idx, doc_id in enumerate(ids_to_delete, 1):
                        ibm_db.bind_param(stmt, idx, doc_id)  # type: ignore[arg-type]
                    ibm_db.execute(stmt)  # type: ignore[arg-type]

            if not batch:
                continue

            # Choose insertion strategy based on use_batch_insert flag
            if self.use_batch_insert:
                written += self._write_documents_batch(batch, policy, conn)
            else:
                written += self._write_documents_individually(batch, policy, conn)

        logger.info(f"Written {written} documents")
        return written

    def _write_documents_batch(
        self, documents: list[Document], policy: DuplicatePolicy, conn: Any
    ) -> int:
        """
        Write documents using multi-row INSERT for better performance.
        
        :param documents: Documents to write.
        :param policy: Duplicate handling policy.
        :param conn: Database connection.
        :return: Number of documents written.
        """
        try:
            # Convert all documents and validate
            batch_data = []
            for doc in documents:
                db2_doc = document_to_db2_dict(doc, self.embedding_dimension)
                batch_data.append((
                    doc.id,
                    doc.content or "",
                    cast(str, db2_doc["meta"]),
                    db2_doc["embedding"]
                ))

            # Build multi-row INSERT SQL
            insert_sql, params = self.query_builder.build_insert_documents_batch(batch_data)
            
            # Prepare and execute
            stmt = ibm_db.prepare(conn, insert_sql)
            if not stmt:
                msg = f"Failed to prepare batch insert: {ibm_db.stmt_error()}"
                raise DocumentStoreError(msg)

            # Bind all parameters
            for idx, param in enumerate(params, 1):
                ibm_db.bind_param(stmt, idx, param)  # type: ignore[arg-type]

            result = ibm_db.execute(stmt)  # type: ignore[arg-type]
            if not result:
                msg = f"Failed to execute batch insert: {ibm_db.stmt_error(stmt)}"  # type: ignore[arg-type]
                raise DocumentStoreError(msg)

            return len(documents)

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            error_msg = str(e)
            # Check for DB2 duplicate key error
            if "SQL0803N" in error_msg or "duplicate" in error_msg.lower():
                if policy == DuplicatePolicy.FAIL:
                    msg = f"Duplicate documents in batch"
                    raise DuplicateDocumentError(msg) from e
                elif policy == DuplicatePolicy.SKIP or policy == DuplicatePolicy.OVERWRITE:
                    # Fall back to individual inserts for granular handling
                    logger.warning("Batch insert failed due to duplicates, falling back to individual inserts")
                    return self._write_documents_individually(documents, policy, conn)
                else:
                    msg = f"Duplicate documents in batch"
                    raise ValueError(msg) from e

            msg = f"Error writing batch: {e}"
            logger.error(msg, exc_info=True)
            raise DocumentStoreError(msg) from e

    def _write_documents_individually(
        self, documents: list[Document], policy: DuplicatePolicy, conn: Any
    ) -> int:
        """
        Fallback method to write documents one-by-one.
        
        Used when batch insertion fails due to duplicates or other errors.
        
        :param documents: Documents to write.
        :param policy: Duplicate handling policy.
        :param conn: Database connection.
        :return: Number of documents written.
        """
        written = 0
        
        for doc in documents:
            try:
                # Use converter for validation and conversion
                db2_doc = document_to_db2_dict(doc, self.embedding_dimension)
                embedding_str = db2_doc["embedding"]

                # Use QueryBuilder for SQL construction
                insert_sql, _ = self.query_builder.build_insert_document(embedding_str)

                stmt = ibm_db.prepare(conn, insert_sql)
                if not stmt:
                    msg = f"Failed to prepare statement: {ibm_db.stmt_error()}"
                    raise DocumentStoreError(msg)

                # Bind parameters
                ibm_db.bind_param(stmt, 1, doc.id)  # type: ignore[arg-type]
                ibm_db.bind_param(stmt, 2, doc.content or "")  # type: ignore[arg-type]
                ibm_db.bind_param(stmt, 3, cast(str, db2_doc["meta"]))  # type: ignore[arg-type]

                result = ibm_db.execute(stmt)  # type: ignore[arg-type]
                if not result:
                    msg = f"Failed to execute statement: {ibm_db.stmt_error(stmt)}"  # type: ignore[arg-type]
                    raise DocumentStoreError(msg)

                written += 1

            except Exception as e:
                error_msg = str(e)
                # Check for DB2 duplicate key error (SQL0803N)
                if "SQL0803N" in error_msg or "duplicate" in error_msg.lower():
                    if policy == DuplicatePolicy.FAIL:
                        msg = f"Document {doc.id} already exists"
                        raise DuplicateDocumentError(msg) from e
                    elif policy == DuplicatePolicy.SKIP or policy == DuplicatePolicy.OVERWRITE:
                        # Skip this document
                        continue
                    else:
                        # NONE raises ValueError for duplicates
                        msg = f"Document {doc.id} already exists"
                        raise ValueError(msg) from e

                msg = f"Error writing document {doc.id}: {e}"
                logger.error(msg, exc_info=True)
                raise DocumentStoreError(msg) from e
        
        return written

    def filter_documents(
        self,
        filters: dict[str, Any] | None = None,
        offset: int | None = None,
        limit: int | None = None,
        return_embedding: bool = False,
    ) -> list[Document]:
        """
        Filter documents with optional pagination.

        Supports Haystack filter format with operators: ==, !=, >, >=, <, <=, in, not in
        and logical operators: AND, OR, NOT.

        :param filters: Filter dict in Haystack format.
        :param offset: Number of documents to skip (for pagination).
        :param limit: Maximum number of documents to return (for pagination).
        :param return_embedding: Whether to include embeddings in returned documents.
        :return: Matching documents.
        """
        conn = self._get_connection()
        where_clause = None
        params: list[Any] = []

        if filters:
            # Use the advanced filter system from filters.py
            where_clause, params = convert_filters(filters)
            # Remove leading " WHERE " if present
            if where_clause.startswith(" WHERE "):
                where_clause = where_clause[7:]

        # Use QueryBuilder for consistent SQL generation with pagination
        sql = self.query_builder.build_select_by_filters(where_clause=where_clause, limit=limit, offset=offset)

        try:
            # Use proper parameter binding instead of string replacement
            if params:
                stmt = ibm_db.prepare(conn, sql)
                if not stmt:
                    msg = f"Failed to prepare statement: {ibm_db.conn_errormsg(conn)}"
                    raise DocumentStoreError(msg)

                for i, param in enumerate(params, 1):
                    ibm_db.bind_param(stmt, i, param)  # type: ignore[arg-type]

                ibm_db.execute(stmt)  # type: ignore[arg-type]
            else:
                stmt = ibm_db.exec_immediate(conn, sql)

            documents = []
            while True:
                row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
                if row is False or not isinstance(row, dict):
                    break
                # Use converter for consistent document creation
                doc = db2_row_to_document(cast(dict[str, Any], row), include_embedding=return_embedding)
                documents.append(doc)

            return documents

        except Exception as e:
            msg = f"Error filtering: {e}"
            raise DocumentStoreError(msg) from e

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by ID.

        :param document_ids: IDs to delete.
        """
        if not document_ids:
            return

        conn = self._get_connection()
        # Use QueryBuilder for consistent SQL generation
        sql, _ = self.query_builder.build_delete_by_ids(len(document_ids))

        try:
            stmt = ibm_db.prepare(conn, sql)
            for i, doc_id in enumerate(document_ids, 1):
                ibm_db.bind_param(stmt, i, doc_id)  # type: ignore[arg-type]
            ibm_db.execute(stmt)  # type: ignore[arg-type]
            logger.info(f"Deleted {len(document_ids)} documents")
        except Exception as e:
            msg = f"Error deleting: {e}"
            raise DocumentStoreError(msg) from e

    def get_documents_by_ids(
        self,
        document_ids: list[str],
        return_embedding: bool = False,
    ) -> list[Document]:
        """
        Get documents by their IDs.

        :param document_ids: List of document IDs to retrieve.
        :param return_embedding: Whether to include embeddings in returned documents.
        :return: List of documents matching the IDs.
        """
        if not document_ids:
            return []

        conn = self._get_connection()
        placeholders = ",".join(["?"] * len(document_ids))
        sql = f"SELECT id, content, embedding, meta FROM {self.qualified_table_name} WHERE id IN ({placeholders})"

        try:
            stmt = ibm_db.prepare(conn, sql)
            if not stmt:
                msg = f"Failed to prepare statement: {ibm_db.conn_errormsg(conn)}"
                raise DocumentStoreError(msg)

            for i, doc_id in enumerate(document_ids, 1):
                ibm_db.bind_param(stmt, i, doc_id)  # type: ignore[arg-type]

            ibm_db.execute(stmt)  # type: ignore[arg-type]

            documents = []
            while True:
                row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
                if row is False or not isinstance(row, dict):
                    break
                # Use converter for consistent document creation
                doc = db2_row_to_document(cast(dict[str, Any], row), include_embedding=return_embedding)
                documents.append(doc)

            return documents

        except Exception as e:
            msg = f"Error retrieving documents by IDs: {e}"
            raise DocumentStoreError(msg) from e

    def update_documents(self, documents: list[Document]) -> None:
        """
        Update existing documents in the store.

        Updates content, metadata, and embeddings for documents with matching IDs.
        Documents that don't exist will be skipped.

        :param documents: Documents to update (must have valid IDs).
        :raises ValueError: If any document is missing an ID or embedding.
        """
        if not documents:
            return

        conn = self._get_connection()
        updated = 0

        for doc in documents:
            try:
                # Validate and convert document
                db2_doc = document_to_db2_dict(doc, self.embedding_dimension)

                # Use QueryBuilder for consistent SQL generation
                update_sql, _ = self.query_builder.build_update_document(db2_doc["embedding"])

                stmt = ibm_db.prepare(conn, update_sql)
                if not stmt:
                    msg = f"Failed to prepare statement: {ibm_db.conn_errormsg(conn)}"
                    raise DocumentStoreError(msg)

                ibm_db.bind_param(stmt, 1, doc.content or "")  # type: ignore[arg-type]
                ibm_db.bind_param(stmt, 2, cast(str, db2_doc["meta"]))  # type: ignore[arg-type]
                ibm_db.bind_param(stmt, 3, doc.id)  # type: ignore[arg-type]

                result = ibm_db.execute(stmt)  # type: ignore[arg-type]
                if result:
                    updated += 1

            except ValueError:
                # Re-raise validation errors
                raise
            except Exception as e:
                msg = f"Error updating document {doc.id}: {e}"
                logger.error(msg, exc_info=True)
                raise DocumentStoreError(msg) from e

        logger.info(f"Updated {updated} documents")

    def delete_by_filter(self, filters: dict[str, Any]) -> None:
        """
        Delete documents matching the given filters.

        This is the Haystack standard method name for deleting documents by filter.

        :param filters: Filter dictionary in Haystack format.
        :raises DocumentStoreError: If filters are invalid or deletion fails.
        """
        if not filters:
            msg = "Filters must be provided for delete_by_filter"
            raise DocumentStoreError(msg)

        conn = self._get_connection()

        # Convert filters to WHERE clause
        where_clause, params = convert_filters(filters)

        # Remove leading " WHERE " if present
        if where_clause.startswith(" WHERE "):
            where_clause = where_clause[7:]

        # Use QueryBuilder for consistent SQL generation
        sql = self.query_builder.build_delete_by_filters(where_clause)

        try:
            # Use proper parameter binding
            if params:
                stmt = ibm_db.prepare(conn, sql)
                if not stmt:
                    msg = f"Failed to prepare statement: {ibm_db.conn_errormsg(conn)}"
                    raise DocumentStoreError(msg)

                for i, param in enumerate(params, 1):
                    ibm_db.bind_param(stmt, i, param)  # type: ignore[arg-type]

                ibm_db.execute(stmt)  # type: ignore[arg-type]
            else:
                ibm_db.exec_immediate(conn, sql)

            logger.info(f"Deleted documents matching filters: {filters}")
        except Exception as e:
            msg = f"Error deleting by filters: {e}"
            raise DocumentStoreError(msg) from e

    # Alias for backwards compatibility
    def delete_documents_by_filters(self, filters: dict[str, Any]) -> None:
        """
        Alias for delete_by_filter() for backwards compatibility.

        :param filters: Filter dictionary in Haystack format.
        """
        return self.delete_by_filter(filters)

    def delete_table(self) -> None:
        """
        Delete the document table from the database.

        This is a destructive operation that removes all documents and the table structure.
        Use with caution.
        """
        self._drop_table_if_exists()
        logger.info(f"Table {self.table_name} deleted")

    def delete_all_documents(self) -> None:
        """
        Delete all documents from the store.

        This removes all documents but keeps the table structure intact.
        """
        conn = self._get_connection()
        try:
            ibm_db.exec_immediate(conn, f"DELETE FROM {self.qualified_table_name}")
            logger.info("All documents deleted")
        except Exception as e:
            msg = f"Error deleting all documents: {e}"
            raise DocumentStoreError(msg) from e

    def count_documents_by_filter(self, filters: dict[str, Any] | None = None) -> int:
        """
        Count documents matching the given filters.

        :param filters: Filter dictionary.
        :return: Number of matching documents.
        """
        conn = self._get_connection()
        sql = f"SELECT COUNT(*) as cnt FROM {self.qualified_table_name}"
        params: list[Any] = []

        if filters:
            # Use convert_filters for consistent filter handling
            where_clause, filter_params = convert_filters(filters)
            if where_clause:
                sql += f" WHERE {where_clause}"
                params = filter_params

        try:
            stmt = ibm_db.prepare(conn, sql)
            for i, param in enumerate(params, 1):
                ibm_db.bind_param(stmt, i, param)  # type: ignore[arg-type]
            ibm_db.execute(stmt)  # type: ignore[arg-type]

            row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
            if row is False or not isinstance(row, dict):
                return 0
            return int(cast(Any, row["CNT"]))
        except Exception as e:
            msg = f"Error counting documents by filter: {e}"
            raise DocumentStoreError(msg) from e

    def get_metadata_fields_info(self) -> dict[str, Any]:
        """
        Get information about metadata fields in the document store.

        Returns a dictionary with field names as keys and field info as values.
        For DB2, this returns basic type information based on JSON metadata.

        :return: Dictionary of metadata field information.
        :raises DocumentStoreError: If query execution fails.
        """
        conn = self._get_connection()

        # Get all unique metadata keys from documents
        sql = f"SELECT DISTINCT meta FROM {self.qualified_table_name} WHERE meta IS NOT NULL"

        try:
            stmt = ibm_db.exec_immediate(conn, sql)
            fields_info: dict[str, dict[str, str]] = {"content": {"type": "text"}}

            while True:
                row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
                if row is False or not isinstance(row, dict):
                    break
                if row.get("META"):
                    try:
                        meta = json.loads(cast(str, row["META"]))
                        for key, value in meta.items():
                            if key not in fields_info:
                                # Determine type based on value
                                if isinstance(value, bool):
                                    field_type = "boolean"
                                elif isinstance(value, int):
                                    field_type = "integer"
                                elif isinstance(value, float):
                                    field_type = "float"
                                else:
                                    field_type = "string"
                                fields_info[key] = {"type": field_type}
                    except json.JSONDecodeError:
                        continue

            return fields_info
        except Exception as e:
            msg = f"Error getting metadata fields info: {e}"
            raise DocumentStoreError(msg) from e

    def get_metadata_field_unique_values(self, field: str) -> list[Any]:
        """
        Get unique values for a specific metadata field.

        :param field: Metadata field name.
        :return: List of unique values.
        :raises DocumentStoreError: If query execution fails.
        """
        conn = self._get_connection()
        sql = f"SELECT DISTINCT JSON_VALUE(meta, '$.{field}') as value FROM {self.qualified_table_name} WHERE JSON_VALUE(meta, '$.{field}') IS NOT NULL"

        try:
            stmt = ibm_db.exec_immediate(conn, sql)
            values = []

            while True:
                row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
                if row is False or not isinstance(row, dict):
                    break
                if row.get("VALUE"):
                    values.append(cast(Any, row["VALUE"]))

            return values
        except Exception as e:
            msg = f"Error getting unique values for field {field}: {e}"
            raise DocumentStoreError(msg) from e

    def get_metadata_field_min_max(self, field: str) -> tuple[Any, Any]:
        """
        Get minimum and maximum values for a numeric metadata field.

        :param field: Metadata field name.
        :return: Tuple of (min_value, max_value).
        :raises ValueError: If field doesn't exist or is not numeric.
        """
        conn = self._get_connection()

        # Check if field exists
        check_sql = (
            f"SELECT JSON_VALUE(meta, '$.{field}') as value "
            f"FROM {self.qualified_table_name} "
            f"WHERE JSON_VALUE(meta, '$.{field}') IS NOT NULL "
            "FETCH FIRST 1 ROWS ONLY"
        )

        try:
            stmt = ibm_db.exec_immediate(conn, check_sql)
            if not ibm_db.fetch_assoc(stmt):  # type: ignore[arg-type]
                msg = f"Field '{field}' not found in document store"
                raise DocumentStoreError(msg)
        except Exception as e:
            msg = f"Field '{field}' not found in document store"
            raise DocumentStoreError(msg) from e

        # Get min and max
        sql = f"""
        SELECT
            MIN(CAST(JSON_VALUE(meta, '$.{field}') AS DECIMAL)) as min_value,
            MAX(CAST(JSON_VALUE(meta, '$.{field}') AS DECIMAL)) as max_value
        FROM {self.qualified_table_name}
        WHERE JSON_VALUE(meta, '$.{field}') IS NOT NULL
        """

        try:
            stmt = ibm_db.exec_immediate(conn, sql)
            row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]

            if row is not False and isinstance(row, dict):
                return (float(cast(Any, row["MIN_VALUE"])), float(cast(Any, row["MAX_VALUE"])))
            return (None, None)
        except Exception as e:
            msg = f"Error getting min/max for field {field}: {e}"
            raise DocumentStoreError(msg) from e

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dict.

        Uses Secret objects to securely handle credentials during serialization.
        """
        return default_to_dict(
            self,
            connection_string=self._connection_string_secret,
            database=self._database,
            username=self._username,
            password=self._password,
            hostname=self._hostname,
            port=self._port,
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
            distance_metric=self.distance_metric,
            embedding_model=self.embedding_model,
            validate_embedding_model=self.validate_embedding_model,
            batch_size=self.batch_size,
            use_ssl=self._use_ssl,
            ssl_certificate=self._ssl_certificate,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DB2DocumentStore":
        """
        Deserialize from dict.

        Properly handles Secret objects for secure credential management.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["connection_string", "username", "password"])
        return default_from_dict(cls, data)

    def __del__(self) -> None:
        """
        Close database connections on deletion.

        Note: With thread-local connections, each thread's connection
        is closed when the thread exits. This method is kept for
        compatibility but has limited effect with thread-local storage.
        """
        # Thread-local connections are automatically cleaned up per thread
        # We can't reliably close all thread connections from here

    # Async method stubs - Not yet implemented
    # These methods are placeholders for future async support

    async def count_documents_async(self) -> int:
        """
        Async version of count_documents.

        .. note::
           Async operations are not yet supported for DB2DocumentStore. All async methods will raise NotImplementedError.
        """
        msg = "Async operations are not yet supported for DB2DocumentStore. Use synchronous methods instead."
        raise NotImplementedError(msg)

    async def write_documents_async(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Async version of write_documents.

        .. note::
           Async operations are not yet supported for DB2DocumentStore. All async methods will raise NotImplementedError.
        """
        msg = "Async operations are not yet supported for DB2DocumentStore. Use synchronous methods instead."
        raise NotImplementedError(msg)

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Async version of delete_documents.

        .. note::
           Async operations are not yet supported for DB2DocumentStore. All async methods will raise NotImplementedError.
        """
        msg = "Async operations are not yet supported for DB2DocumentStore. Use synchronous methods instead."
        raise NotImplementedError(msg)

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Async version of filter_documents.

        .. note::
           Async operations are not yet supported for DB2DocumentStore. All async methods will raise NotImplementedError.
        """
        msg = "Async operations are not yet supported for DB2DocumentStore. Use synchronous methods instead."
        raise NotImplementedError(msg)
        """
        Async version of filter_documents.

        :raises NotImplementedError: Async operations not yet supported.
        """
        msg = "Async operations are not yet supported for DB2DocumentStore. Use synchronous methods instead."
        raise NotImplementedError(msg)

    def query_by_embedding(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        return_embedding: bool = False,
        offset: int = 0,
    ) -> list[Document]:
        """
        Find documents most similar to the given embedding using vector similarity search.

        This is the public API for embedding-based retrieval, providing direct access
        to vector similarity search without requiring a retriever component.

        :param query_embedding: Query embedding vector.
        :param filters: Optional filters to narrow the search space.
        :param top_k: Maximum number of documents to return.
        :param return_embedding: If True, include embeddings in returned documents.
        :param offset: Number of documents to skip (for pagination). Default is 0.
        :return: List of documents ordered by similarity (most similar first).
        :raises ValueError: If query_embedding dimension doesn't match store dimension.
        :raises DocumentStoreError: If the query fails.

        Usage example:
        ```python
        from haystack_integrations.document_stores.db2 import DB2DocumentStore
        from haystack.utils import Secret

        store = DB2DocumentStore(
            database="TESTDB",
            hostname="db2.example.com",
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            embedding_dimension=384
        )

        # Get embedding from your embedder
        query_embedding = [0.1, 0.2, ...] # 384-dim vector

        # Query directly
        results = store.query_by_embedding(
            query_embedding=query_embedding,
            filters={"category": "technology"},
            top_k=5
        )

        for doc in results:
            print(f"Score: {doc.score}, Content: {doc.content}")

        # Pagination example - get next page
        next_page = store.query_by_embedding(
            query_embedding=query_embedding,
            filters={"category": "technology"},
            top_k=5,
            offset=5  # Skip first 5 results
        )
        ```
        """
        return self._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            return_embedding=return_embedding,
            offset=offset,
        )

    def query_by_keyword(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Find documents matching the given keyword query.

        This is the public API for keyword-based retrieval, providing direct access
        to keyword search without requiring a retriever component.

        Uses multi-word keyword matching with scoring based on the number of
        matching keywords. All keywords must be present in the document.

        :param query: Query string (can contain multiple keywords).
        :param filters: Optional filters to narrow the search space.
        :param top_k: Maximum number of documents to return.
        :return: List of documents ordered by keyword match score (best matches first).
        :raises DocumentStoreError: If the query fails.

        Usage example:
        ```python
        from haystack_integrations.document_stores.db2 import DB2DocumentStore
        from haystack.utils import Secret

        import os

        use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
        port = int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000"))

        store = DB2DocumentStore(
            database=os.getenv("DB2_DATABASE", "TESTDB"),
            hostname=os.getenv("DB2_HOSTNAME"),
            port=port,
            username=Secret.from_env_var("DB2_USER"),
            password=Secret.from_env_var("DB2_PASSWORD"),
            embedding_dimension=384,
            use_ssl=use_ssl,
        )

        # Query with keywords
        results = store.query_by_keyword(
            query="machine learning algorithms",
            filters={"category": "technology"},
            top_k=5
        )

        for doc in results:
            print(f"Score: {doc.score}, Content: {doc.content}")
        ```
        """
        return self._keyword_retrieval(
            query=query,
            filters=filters,
            top_k=top_k,
        )

        # This is acceptable as connections close when threads terminate

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        return_embedding: bool = False,
        offset: int = 0,
    ) -> list[Document]:
        """
        Internal method for embedding-based retrieval.

        Used by DB2HybridRetriever to perform vector similarity search.

        :param query_embedding: Query embedding vector.
        :param top_k: Number of documents to retrieve.
        :param filters: Optional filters to apply.
        :param return_embedding: Whether to include embeddings in returned documents.
        :param offset: Number of documents to skip (for pagination).
        :return: Retrieved documents with similarity scores.
        """
        if len(query_embedding) != self.embedding_dimension:
            msg = (
                f"Query embedding dimension {len(query_embedding)} "
                f"doesn't match store dimension {self.embedding_dimension}"
            )
            raise ValueError(msg)

        conn = self._get_connection()
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Build query with optional filters
        where_clause = None
        params: list[Any] = []
        if filters:
            where_clause, params = convert_filters(filters)

        # Use QueryBuilder for vector search with pagination support
        # Cast distance_metric to proper type for type checker
        sql = self.query_builder.build_vector_search(
            distance_metric=cast(Literal["cosine", "euclidean", "dot"], self.distance_metric),
            embedding_str=embedding_str,
            top_k=top_k,
            where_clause=where_clause,
            offset=offset,
        )

        # # TEMPORARY: Log SQL query with truncated vector for debugging
        # truncated_sql = sql.replace(embedding_str, f"[...{len(query_embedding)} dimensions...]")
        # logger.info(f"\n{'='*80}\nVECTOR SEARCH SQL:\n{truncated_sql}\n{'='*80}\n")

        try:
            # Use proper parameter binding for filters
            if params:
                stmt = ibm_db.prepare(conn, sql)
                if not stmt:
                    msg = f"Failed to prepare statement: {ibm_db.conn_errormsg(conn)}"
                    raise DocumentStoreError(msg)

                for i, param in enumerate(params, 1):
                    ibm_db.bind_param(stmt, i, param)  # type: ignore[arg-type]

                ibm_db.execute(stmt)  # type: ignore[arg-type]
            else:
                stmt = ibm_db.exec_immediate(conn, sql)

            documents = []

            while True:
                row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
                if row is False or not isinstance(row, dict):
                    break
                # Use converter for consistent document creation
                doc = db2_row_to_document(cast(dict[str, Any], row), include_embedding=return_embedding)
                # Convert distance to score (lower distance = better, so invert it)
                # For cosine distance (range 0-2): score = 1 - distance
                # This gives: distance 0 (identical) → score 1.0, distance 2 (opposite) → score -1.0
                if row.get("DISTANCE") is not None:
                    distance = float(cast(Any, row["DISTANCE"]))
                    score = 1.0 - distance
                    # Set score on document and in metadata for Haystack compatibility
                    doc = replace(doc, score=score)
                    if doc.meta is None:
                        doc.meta = {}
                    doc.meta["score"] = score
                documents.append(doc)

            return documents
        except Exception as e:
            msg = f"Embedding retrieval failed: {e}"
            raise DocumentStoreError(msg) from e

    def _keyword_retrieval(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Internal method for keyword-based retrieval.

        Used by DB2HybridRetriever to perform keyword search.

        :param query: Query string.
        :param top_k: Number of documents to retrieve.
        :param filters: Optional filters to apply.
        :return: Retrieved documents with keyword scores.
        """
        if not query or not query.strip():
            return []

        conn = self._get_connection()

        # Build query with optional filters
        where_clause = None
        params: list[Any] = []
        if filters:
            where_clause, params = convert_filters(filters)

        # Use QueryBuilder for keyword search
        sql = self.query_builder.build_keyword_search(
            query=query.strip(),
            top_k=top_k,
            where_clause=where_clause,
        )

        try:
            # Use proper parameter binding for filters
            if params:
                stmt = ibm_db.prepare(conn, sql)
                if not stmt:
                    msg = f"Failed to prepare statement: {ibm_db.conn_errormsg(conn)}"
                    raise DocumentStoreError(msg)

                for i, param in enumerate(params, 1):
                    ibm_db.bind_param(stmt, i, param)  # type: ignore[arg-type]

                ibm_db.execute(stmt)  # type: ignore[arg-type]
            else:
                stmt = ibm_db.exec_immediate(conn, sql)

            documents = []

            while True:
                row = ibm_db.fetch_assoc(stmt)  # type: ignore[arg-type]
                if row is False or not isinstance(row, dict):
                    break
                # Use converter for consistent document creation
                doc = db2_row_to_document(row, include_embedding=False)
                # Add keyword score
                if row.get("SCORE") is not None:
                    doc = replace(doc, score=float(cast(Any, row["SCORE"])))
                documents.append(doc)

            return documents
        except Exception as e:
            msg = f"Keyword retrieval failed: {e}"
            raise DocumentStoreError(msg) from e
