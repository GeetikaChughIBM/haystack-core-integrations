# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.document_stores.db2.filters import (
    _format_value_for_json,
    _is_valid_field_name,
    _normalize_filters,
    _parse_comparison_condition,
    _parse_logical_condition,
    _validate_filters,
    convert_filters,
)


def test_format_value_for_json_bool() -> None:
    assert _format_value_for_json(True) == "true"
    assert _format_value_for_json(False) == "false"


def test_format_value_for_json_non_bool() -> None:
    assert _format_value_for_json("Nike") == "Nike"
    assert _format_value_for_json(42) == "42"


def test_is_valid_field_name_accepts_safe_names() -> None:
    assert _is_valid_field_name("brand")
    assert _is_valid_field_name("meta.brand")
    assert _is_valid_field_name("nested.field_name_1")


def test_is_valid_field_name_rejects_unsafe_names() -> None:
    assert not _is_valid_field_name("brand;DROP TABLE users")
    assert not _is_valid_field_name("brand-name")
    assert not _is_valid_field_name("brand value")


def test_normalize_filters_simple_dict() -> None:
    normalized = _normalize_filters({"brand": "Nike"})
    assert normalized == {"field": "brand", "operator": "==", "value": "Nike"}


def test_normalize_filters_operator_dict() -> None:
    normalized = _normalize_filters({"price": {"$gte": 100}})
    assert normalized == {"field": "price", "operator": ">=", "value": 100}


def test_normalize_filters_multiple_fields_become_and() -> None:
    normalized = _normalize_filters({"brand": "Nike", "category": "Shoes"})
    assert normalized == {
        "operator": "AND",
        "conditions": [
            {"field": "brand", "operator": "==", "value": "Nike"},
            {"field": "category", "operator": "==", "value": "Shoes"},
        ],
    }


def test_normalize_filters_logical_operators() -> None:
    normalized = _normalize_filters({"$or": [{"brand": "Nike"}, {"brand": "Adidas"}]})
    assert normalized == {
        "operator": "OR",
        "conditions": [
            {"field": "brand", "operator": "==", "value": "Nike"},
            {"field": "brand", "operator": "==", "value": "Adidas"},
        ],
    }


def test_validate_filters_rejects_non_dict() -> None:
    with pytest.raises(TypeError, match="Filters must be a dictionary"):
        _validate_filters(["not", "a", "dict"])  # type: ignore[arg-type]


def test_validate_filters_rejects_invalid_operator() -> None:
    with pytest.raises(ValueError, match="Invalid operator"):
        _validate_filters({"field": "brand", "operator": "LIKE", "value": "Nike"})


def test_validate_filters_rejects_empty_in_list() -> None:
    with pytest.raises(ValueError, match="requires a non-empty list"):
        _validate_filters({"field": "brand", "operator": "in", "value": []})


def test_validate_filters_rejects_not_with_multiple_conditions() -> None:
    with pytest.raises(ValueError, match="NOT operator must have exactly one condition"):
        _validate_filters(
            {
                "operator": "NOT",
                "conditions": [
                    {"field": "brand", "operator": "==", "value": "Nike"},
                    {"field": "category", "operator": "==", "value": "Shoes"},
                ],
            }
        )


def test_parse_comparison_condition_equality() -> None:
    query, values = _parse_comparison_condition({"field": "brand", "operator": "==", "value": "Nike"})
    assert query == "JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?"
    assert values == ["Nike"]


def test_parse_comparison_condition_strips_meta_prefix() -> None:
    query, values = _parse_comparison_condition({"field": "meta.brand", "operator": "==", "value": "Nike"})
    assert query == "JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?"
    assert values == ["Nike"]


def test_parse_comparison_condition_numeric_comparison() -> None:
    query, values = _parse_comparison_condition({"field": "price", "operator": ">=", "value": 100})
    assert "CAST(CAST(JSON_VALUE(meta, '$.price' RETURNING VARCHAR(1000)) AS VARCHAR(100)) AS DECFLOAT)" in query
    assert ">= CAST(? AS DECFLOAT)" in query
    assert values == [100]


def test_parse_comparison_condition_in_operator() -> None:
    query, values = _parse_comparison_condition({"field": "brand", "operator": "in", "value": ["Nike", "Adidas"]})
    assert query == "JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) IN (?,?)"
    assert values == ["Nike", "Adidas"]


def test_parse_comparison_condition_not_in_operator() -> None:
    query, values = _parse_comparison_condition({"field": "brand", "operator": "not in", "value": ["Nike", "Puma"]})
    assert query == "JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) NOT IN (?,?)"
    assert values == ["Nike", "Puma"]


def test_parse_comparison_condition_bool_value() -> None:
    query, values = _parse_comparison_condition({"field": "available", "operator": "==", "value": True})
    assert query == "JSON_VALUE(meta, '$.available' RETURNING VARCHAR(1000)) = ?"
    assert values == ["true"]


def test_parse_comparison_condition_rejects_invalid_field_name() -> None:
    with pytest.raises(ValueError, match="Invalid field name"):
        _parse_comparison_condition({"field": "brand;DROP TABLE x", "operator": "==", "value": "Nike"})


def test_parse_logical_condition_and() -> None:
    query, values = _parse_logical_condition(
        {
            "operator": "AND",
            "conditions": [
                {"field": "brand", "operator": "==", "value": "Nike"},
                {"field": "category", "operator": "==", "value": "Shoes"},
            ],
        }
    )
    assert query == (
        "(JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?) AND "
        "(JSON_VALUE(meta, '$.category' RETURNING VARCHAR(1000)) = ?)"
    )
    assert values == ["Nike", "Shoes"]


def test_parse_logical_condition_or() -> None:
    query, values = _parse_logical_condition(
        {
            "operator": "OR",
            "conditions": [
                {"field": "brand", "operator": "==", "value": "Nike"},
                {"field": "brand", "operator": "==", "value": "Adidas"},
            ],
        }
    )
    assert query == (
        "(JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?) OR "
        "(JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?)"
    )
    assert values == ["Nike", "Adidas"]


def test_parse_logical_condition_not() -> None:
    query, values = _parse_logical_condition(
        {
            "operator": "NOT",
            "conditions": [{"field": "brand", "operator": "==", "value": "Nike"}],
        }
    )
    assert query == "NOT (JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?)"
    assert values == ["Nike"]


def test_parse_logical_condition_rejects_empty_conditions() -> None:
    with pytest.raises(ValueError, match="at least one condition"):
        _parse_logical_condition({"operator": "AND", "conditions": []})


def test_convert_filters_none() -> None:
    query, values = convert_filters(None)
    assert query == ""
    assert values == []


def test_convert_filters_without_where() -> None:
    query, values = convert_filters({"brand": "Nike"})
    assert query == "JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?"
    assert values == ["Nike"]


def test_convert_filters_with_where() -> None:
    query, values = convert_filters({"brand": "Nike"}, include_where=True)
    assert query == " WHERE JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?"
    assert values == ["Nike"]


def test_convert_filters_nested_structure() -> None:
    query, values = convert_filters(
        {
            "operator": "AND",
            "conditions": [
                {"field": "brand", "operator": "==", "value": "Nike"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "category", "operator": "==", "value": "Shoes"},
                        {"field": "category", "operator": "==", "value": "Running"},
                    ],
                },
            ],
        }
    )
    assert query == (
        "(JSON_VALUE(meta, '$.brand' RETURNING VARCHAR(1000)) = ?) AND "
        "((JSON_VALUE(meta, '$.category' RETURNING VARCHAR(1000)) = ?) OR "
        "(JSON_VALUE(meta, '$.category' RETURNING VARCHAR(1000)) = ?))"
    )
    assert values == ["Nike", "Shoes", "Running"]

# Made with Bob
