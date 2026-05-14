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
def document_store_local(request, monkeypatch):
    """
    Fixture for local DB2 connection (3-parameter).
    Uses environment variables: DB2_USER, DB2_PASSWORD, DB2_DATABASE
    """
    # Check if local DB2 credentials are available
    if not all([os.getenv("DB2_USER"), os.getenv("DB2_PASSWORD"), os.getenv("DB2_DATABASE", "TESTDB")]):
        pytest.skip("Local DB2 credentials not available. Set DB2_USER, DB2_PASSWORD, DB2_DATABASE")

    # Use shorter table name to avoid DB2's 128 char limit
    test_name = request.node.name.replace("test_", "").replace("[local]", "")[:30]
    table_name = f"hs_{test_name}"
    embedding_dimension = 384
    distance_metric = "cosine"
    recreate_table = True

    store = DB2DocumentStore(
        database=os.getenv("DB2_DATABASE", "TESTDB"),
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        distance_metric=distance_metric,
        recreate_table=recreate_table,
    )

    yield store

    # Cleanup
    try:
        store._drop_table_if_exists()
    except Exception:
        pass


@pytest.fixture
def document_store_remote(request, monkeypatch):
    """
    Fixture for remote DB2 connection (connection string or parameters).
    Uses environment variables for IBM Cloud DB2 connection.
    """
    # Check if remote DB2 credentials are available
    db2_host = os.getenv("DB2_REMOTE_HOST")
    db2_database = os.getenv("DB2_REMOTE_DATABASE", "BLUDB")
    db2_port = int(os.getenv("DB2_REMOTE_PORT", "32310"))
    db2_user = os.getenv("DB2_REMOTE_USER")
    db2_password = os.getenv("DB2_REMOTE_PASSWORD")

    if not all([db2_host, db2_user, db2_password]):
        pytest.skip("Remote DB2 credentials not available. Set DB2_REMOTE_HOST, DB2_REMOTE_USER, DB2_REMOTE_PASSWORD")

    table_name = f"haystack_test_{request.node.name}"
    embedding_dimension = 384
    distance_metric = "cosine"
    recreate_table = True

    store = DB2DocumentStore(
        database=db2_database,
        hostname=db2_host,
        port=db2_port,
        username=Secret.from_token(db2_user),
        password=Secret.from_token(db2_password),
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        distance_metric=distance_metric,
        recreate_table=recreate_table,
    )

    yield store

    # Cleanup
    try:
        store._drop_table_if_exists()
    except Exception:
        pass


@pytest.fixture
def document_store_connection_string(request, monkeypatch):
    """
    Fixture for remote DB2 connection using connection string.
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
    )

    yield store

    # Cleanup
    try:
        store._drop_table_if_exists()
    except Exception:
        pass


@pytest.fixture(params=["local", "remote", "connection_string"])
def document_store(request):
    """
    Parametrized fixture that tests all connection types.
    Will skip connection types that don't have credentials available.
    """
    if request.param == "local":
        return request.getfixturevalue("document_store_local")
    elif request.param == "remote":
        return request.getfixturevalue("document_store_remote")
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
        username=Secret.from_env_var("DB2_USER"),
        password=Secret.from_env_var("DB2_PASSWORD"),
        table_name=table_name,
        embedding_dimension=embedding_dimension,
        distance_metric=distance_metric,
        recreate_table=False,
    )

    yield store
