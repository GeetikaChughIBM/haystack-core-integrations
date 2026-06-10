# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from haystack.utils import Secret

from haystack_integrations.document_stores.db2 import DB2DocumentStore

# Load .env file from the db2 integration directory
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


@pytest.fixture
def document_store_env(request, monkeypatch):
    """
    Fixture for DB2 connection flow.
    Uses environment variables from `.env` or the current shell.
    """
    db2_database = os.getenv("DB2_DATABASE", "TESTDB")
    db2_host = os.getenv("DB2_HOSTNAME")
    use_ssl = os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"}
    db2_port = int(os.getenv("DB2_SSL_PORT", "50001")) if use_ssl else int(os.getenv("DB2_PORT", "50000"))
    db2_user = os.getenv("DB2_USER")
    db2_password = os.getenv("DB2_PASSWORD")

    if not all([db2_database, db2_user, db2_password]):
        pytest.skip("DB2 credentials not available. Set DB2_DATABASE, DB2_USER, and DB2_PASSWORD in .env")

    raw_test_name = request.node.name.replace("test_", "")
    sanitized_test_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw_test_name)
    table_name = f"hs_{sanitized_test_name[:26]}"
    embedding_dimension = 384
    distance_metric = "cosine"
    recreate_table = True

    store = DB2DocumentStore(
        database=db2_database,
        hostname=db2_host,
        port=db2_port,
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        distance_metric=distance_metric,
        recreate_table=recreate_table,
        use_ssl=use_ssl,
        ssl_certificate=os.getenv("DB2_SSL_CERTIFICATE") or os.getenv("DB2_SSL_CERT_PATH"),
    )

    yield store

    try:
        store._drop_table_if_exists()
    except Exception:
        pass


@pytest.fixture
def document_store_connection_string(request, monkeypatch):
    """
    Fixture for the environment-driven DB2 connection string flow.
    """
    connection_string = os.getenv("DB2_CONNECTION_STRING")

    if not connection_string:
        pytest.skip("DB2_CONNECTION_STRING not set")

    table_name = f"haystack_test_{request.node.name}"
    embedding_dimension = 384
    distance_metric = "cosine"
    recreate_table = True

    store = DB2DocumentStore(
        connection_string=Secret.from_token(connection_string),
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        distance_metric=distance_metric,
        recreate_table=recreate_table,
        use_ssl=os.getenv("DB2_SSL_ENABLED", "").lower() in {"1", "true", "yes"},
        ssl_certificate=os.getenv("DB2_SSL_CERTIFICATE") or os.getenv("DB2_SSL_CERT_PATH"),
    )

    yield store

    try:
        store._drop_table_if_exists()
    except Exception:
        pass


@pytest.fixture(params=["env", "connection_string"])
def document_store(request):
    """
    Parametrized fixture that tests the supported DB2 connection flows.
    """
    if request.param == "env":
        return request.getfixturevalue("document_store_env")
    elif request.param == "connection_string":
        return request.getfixturevalue("document_store_connection_string")


@pytest.fixture
def patches_for_unit_tests():
    """
    Mock patches for unit tests that don't require actual DB2 connection.
    """
    with (
        patch("ibm_db.connect") as mock_connect,
        patch("ibm_db.exec_immediate") as mock_exec,
        patch("ibm_db.prepare") as mock_prepare,
    ):
        # Setup mock connection
        mock_conn = object()
        mock_connect.return_value = mock_conn

        yield mock_connect, mock_exec, mock_prepare


@pytest.fixture
def mock_store(_patches_for_unit_tests, monkeypatch):
    """
    Mock document store for unit tests without DB2 connection.
    """
    monkeypatch.setenv("DB2_USER", "test_user")
    monkeypatch.setenv("DB2_PASSWORD", "test_password")

    table_name = "haystack_test"
    embedding_dimension = 384
    distance_metric = "cosine"

    store = DB2DocumentStore(
        database="TESTDB",
        hostname="db2.example.com",
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        distance_metric=distance_metric,
        recreate_table=False,
    )

    yield store
