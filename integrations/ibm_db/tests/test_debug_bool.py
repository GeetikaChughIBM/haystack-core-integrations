# Temporary diagnostic — DELETE after we learn how DB2 stores/returns JSON booleans.
# Run with:  hatch run test:pytest tests/test_debug_bool.py -s

import pytest
from haystack import Document


@pytest.mark.integration
def test_debug_boolean_storage(document_store):
    """Write a boolean meta value and print how DB2 stores and extracts it."""
    document_store.write_documents(
        [Document(id="d1", content="hello", meta={"updated": True, "name": "foo"}, embedding=[0.1] * 768)]
    )

    table = document_store.table_name
    conn = document_store._get_connection()

    probes = {
        "raw_bson2json": f"SELECT SYSTOOLS.BSON2JSON(meta) FROM {table}",
        "returning_varchar_bool": (
            f"SELECT JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.updated' RETURNING VARCHAR(1000)) FROM {table}"
        ),
        "cast_varchar_bool": (
            f"SELECT CAST(JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.updated') AS VARCHAR(1000)) FROM {table}"
        ),
        "returning_varchar_string": (
            f"SELECT JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.name' RETURNING VARCHAR(1000)) FROM {table}"
        ),
        "json_exists_true": (
            f"SELECT CASE WHEN JSON_EXISTS(SYSTOOLS.BSON2JSON(meta), '$.updated?(@ == true)') "
            f"THEN 1 ELSE 0 END FROM {table}"
        ),
    }

    print("\n===== DB2 BOOLEAN DIAGNOSTIC =====")
    for label, sql in probes.items():
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
            print(f"[{label}] -> {rows!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] -> ERROR: {type(e).__name__}: {e}")
    print("===== END DIAGNOSTIC =====\n")
