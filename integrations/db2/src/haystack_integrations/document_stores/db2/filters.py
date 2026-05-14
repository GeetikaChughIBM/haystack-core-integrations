# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Advanced filter system for DB2 document store.

Converts Haystack filter dictionaries to DB2 SQL WHERE clauses with proper
parameter binding for security and type handling.
"""

import re
from typing import Any, Literal


def _format_value_for_json(value: Any) -> str:
    """
    Format value for JSON comparison in DB2.

    Handles boolean values correctly as JSON stores them as lowercase strings.

    :param value: Value to format.
    :return: Formatted string value.
    """
    if isinstance(value, bool):
        # JSON stores booleans as lowercase 'true'/'false'
        return str(value).lower()
    return str(value)


def _validate_filter_structure(filters: dict[str, Any]) -> None:
    """
    Recursively validate filter dictionary structure.

    :param filters: Filter dictionary.
    :raises ValueError: If structure is invalid.
    """
    if "field" in filters:
        # Comparison condition
        if "operator" not in filters or "value" not in filters:
            msg = "Comparison filter must have 'field', 'operator', and 'value'"
            raise ValueError(msg)

        # Validate operator
        valid_operators = ["==", "!=", ">", ">=", "<", "<=", "in", "not in"]
        if filters["operator"] not in valid_operators:
            msg = f"Invalid operator: {filters['operator']}"
            raise ValueError(msg)

        # Validate 'in'/'not in' values
        if filters["operator"] in ["in", "not in"]:
            if not isinstance(filters["value"], list):
                msg = f"'{filters['operator']}' operator requires a list"
                raise ValueError(msg)
            if not filters["value"]:
                msg = f"'{filters['operator']}' operator requires a non-empty list"
                raise ValueError(msg)

    elif "operator" in filters:
        # Logical condition
        if "conditions" not in filters:
            msg = "Logical filter must have 'conditions'"
            raise ValueError(msg)

        if not isinstance(filters["conditions"], list):
            msg = "'conditions' must be a list"
            raise ValueError(msg)

        if not filters["conditions"]:
            msg = "'conditions' cannot be empty"
            raise ValueError(msg)

        # Validate NOT has exactly one condition
        if filters["operator"] == "NOT" and len(filters["conditions"]) != 1:
            msg = "NOT operator must have exactly one condition"
            raise ValueError(msg)

        # Recursively validate nested conditions
        for condition in filters["conditions"]:
            _validate_filter_structure(condition)


def _validate_filters(filters: dict[str, Any] | None = None) -> None:
    """
    Validate the filters provided.

    :param filters: Filter dictionary to validate.
    :raises TypeError: If filters is not a dictionary.
    :raises ValueError: If filter syntax is invalid.
    """
    if filters:
        if not isinstance(filters, dict):
            msg = "Filters must be a dictionary"
            raise TypeError(msg)
        _validate_filter_structure(filters)


def _convert_filters_to_where_clause(
    filters: dict[str, Any],
    operator: Literal["WHERE", "AND", ""] = "WHERE",
) -> tuple[str, list[Any]]:
    """
    Convert Haystack filters to a WHERE clause and parameter list for DB2.

    Supports:
    - Comparison operators: ==, !=, >, >=, <, <=, in, not in
    - Logical operators: AND, OR, NOT
    - Pattern matching: LIKE (using % wildcards)

    :param filters: Haystack filter dictionary.
    :param operator: SQL operator to use (WHERE, AND, or empty string for just the condition).
    :return: Tuple of (SQL WHERE clause string, list of parameters).
    """
    if "field" in filters:
        query, values = _parse_comparison_condition(filters)
    else:
        query, values = _parse_logical_condition(filters)

    if operator:
        where_clause = f" {operator} {query}"
    else:
        where_clause = query
    return where_clause, values


def _parse_comparison_condition(condition: dict[str, Any]) -> tuple[str, list[Any]]:
    """
    Parse a comparison condition into SQL.

    :param condition: Comparison condition dictionary.
    :return: Tuple of (SQL condition string, list of parameters).
    """
    field = condition["field"]
    operator = condition["operator"]
    value = condition["value"]

    # Validate field name to prevent SQL injection
    if not _is_valid_field_name(field):
        msg = f"Invalid field name: {field}"
        raise ValueError(msg)

    # Strip "meta." prefix if present since we're already accessing the meta column
    if field.startswith("meta."):
        field = field[5:]  # Remove "meta." prefix

    # Build JSON path for metadata field with error handling
    # Use RETURNING clause to ensure proper type handling
    json_path = f"JSON_VALUE(meta, '$.{field}' RETURNING VARCHAR(1000))"

    # Handle different operators
    if operator == "==":
        return f"{json_path} = ?", [_format_value_for_json(value)]
    elif operator == "!=":
        return f"{json_path} != ?", [_format_value_for_json(value)]
    elif operator == ">":
        # Cast both sides to DECFLOAT for numeric comparison
        # DB2 requires explicit type path: JSON_VALUE returns VARCHAR, which we cast to DECFLOAT
        # Also cast the comparison value to ensure type compatibility
        return f"CAST(CAST({json_path} AS VARCHAR(100)) AS DECFLOAT) > CAST(? AS DECFLOAT)", [value]
    elif operator == ">=":
        return f"CAST(CAST({json_path} AS VARCHAR(100)) AS DECFLOAT) >= CAST(? AS DECFLOAT)", [value]
    elif operator == "<":
        return f"CAST(CAST({json_path} AS VARCHAR(100)) AS DECFLOAT) < CAST(? AS DECFLOAT)", [value]
    elif operator == "<=":
        return f"CAST(CAST({json_path} AS VARCHAR(100)) AS DECFLOAT) <= CAST(? AS DECFLOAT)", [value]
    elif operator == "in":
        if not isinstance(value, list):
            msg = f"Value for 'in' operator must be a list, got {type(value).__name__}"
            raise ValueError(msg)
        if not value:
            msg = "Value for 'in' operator cannot be an empty list"
            raise ValueError(msg)
        placeholders = ",".join(["?"] * len(value))
        return f"{json_path} IN ({placeholders})", [_format_value_for_json(v) for v in value]
    elif operator == "not in":
        if not isinstance(value, list):
            msg = f"Value for 'not in' operator must be a list, got {type(value).__name__}"
            raise ValueError(msg)
        if not value:
            msg = "Value for 'not in' operator cannot be an empty list"
            raise ValueError(msg)
        placeholders = ",".join(["?"] * len(value))
        return f"{json_path} NOT IN ({placeholders})", [_format_value_for_json(v) for v in value]
    else:
        msg = f"Unsupported operator: {operator}"
        raise ValueError(msg)


def _parse_logical_condition(condition: dict[str, Any]) -> tuple[str, list[Any]]:
    """
    Parse a logical condition (AND, OR, NOT) into SQL.

    :param condition: Logical condition dictionary.
    :return: Tuple of (SQL condition string, list of parameters).
    """
    operator = condition.get("operator", "AND").upper()
    conditions = condition.get("conditions", [])

    if not conditions:
        msg = "Logical operator must have at least one condition"
        raise ValueError(msg)

    if operator == "NOT":
        if len(conditions) != 1:
            msg = "NOT operator must have exactly one condition"
            raise ValueError(msg)
        sub_query, sub_values = _convert_filters_to_where_clause(conditions[0], operator="AND")
        # Remove the leading " AND " from sub_query
        sub_query = sub_query.strip().replace("AND ", "", 1).replace("WHERE ", "", 1)
        return f"NOT ({sub_query})", sub_values

    # Handle AND/OR
    sub_queries = []
    all_values = []

    for sub_condition in conditions:
        if "field" in sub_condition:
            sub_query, sub_values = _parse_comparison_condition(sub_condition)
        else:
            sub_query, sub_values = _parse_logical_condition(sub_condition)

        sub_queries.append(f"({sub_query})")
        all_values.extend(sub_values)

    combined_query = f" {operator} ".join(sub_queries)
    return combined_query, all_values


def _is_valid_field_name(field: str) -> bool:
    """
    Validate field name to prevent SQL injection.

    Field names should only contain alphanumeric characters, underscores, and dots.

    :param field: Field name to validate.
    :return: True if valid, False otherwise.
    """
    # Allow alphanumeric, underscore, and dot for nested fields
    pattern = r"^[a-zA-Z0-9_.]+$"
    return bool(re.match(pattern, field))


def _normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize simple dict filters to Haystack standard format.

    Supports multiple formats:
    - Simple equality: {'key': 'value'} -> {"field": "key", "operator": "==", "value": "value"}
    - Operator dict: {'price': {'$gte': 100}} -> {"field": "price", "operator": ">=", "value": 100}
    - Logical operators: {'$and': [...]} -> {"operator": "AND", "conditions": [...]}

    :param filters: Filter dictionary (simple or Haystack format).
    :return: Normalized Haystack filter dictionary.
    """
    # Check if already in Haystack format (has 'operator' or 'field' keys)
    if "operator" in filters or "field" in filters:
        return filters

    # Check for logical operators ($and, $or, $not)
    if "$and" in filters:
        return {"operator": "AND", "conditions": [_normalize_filters(c) for c in filters["$and"]]}
    if "$or" in filters:
        return {"operator": "OR", "conditions": [_normalize_filters(c) for c in filters["$or"]]}
    if "$not" in filters:
        return {"operator": "NOT", "conditions": [_normalize_filters(filters["$not"])]}

    # Convert field filters
    conditions = []
    for key, value in filters.items():
        if isinstance(value, dict):
            # Handle operator dict like {'$gte': 100, '$lte': 200}
            for op_key, op_value in value.items():
                operator_map = {
                    "$eq": "==",
                    "$ne": "!=",
                    "$gt": ">",
                    "$gte": ">=",
                    "$lt": "<",
                    "$lte": "<=",
                    "$in": "in",
                    "$nin": "not in",
                }
                operator = operator_map.get(op_key, "==")
                conditions.append({"field": key, "operator": operator, "value": op_value})
        else:
            # Simple equality
            conditions.append({"field": key, "operator": "==", "value": value})

    if len(conditions) == 1:
        return conditions[0]
    elif len(conditions) > 1:
        return {"operator": "AND", "conditions": conditions}

    return filters


def convert_filters(filters: dict[str, Any] | None, include_where: bool = False) -> tuple[str, list[Any]]:
    """
    Convert Haystack filters to DB2 SQL WHERE clause.

    Public API for filter conversion. Supports both:
    - Simple dict format: {'brand': 'Nike', 'price': 100}
    - Haystack standard format: {"field": "brand", "operator": "==", "value": "Nike"}

    :param filters: Filter dictionary or None.
    :param include_where: If True, includes "WHERE" keyword in output. Default False.
    :return: Tuple of (SQL condition clause, list of parameters).
             Returns ("", []) if filters is None.
    """
    if not filters:
        return "", []

    # Normalize simple dict filters to Haystack format
    normalized_filters = _normalize_filters(filters)

    _validate_filters(normalized_filters)

    if include_where:
        return _convert_filters_to_where_clause(normalized_filters, operator="WHERE")
    else:
        # Return just the condition without WHERE keyword
        return _convert_filters_to_where_clause(normalized_filters, operator="")
