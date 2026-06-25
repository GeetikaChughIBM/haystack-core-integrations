# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, ClassVar

_RANGE_OPS = {">", ">=", "<", "<="}


class FilterTranslator:
    """
    Translates Haystack 2.x filter dicts into IBM DB2 SQL WHERE fragments.

    Example input:
        {"operator": "AND", "conditions": [
            {"field": "meta.author", "operator": "==", "value": "Alice"},
            {"field": "meta.year",   "operator": ">",  "value": 2020},
        ]}

    Example output SQL fragment:
        (CAST(JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.author') AS VARCHAR(1000)) = ?
         AND CAST(JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.year') AS VARCHAR(1000)) > ?)

    Params list is built in-place; caller uses it for cursor.execute bindings.
    """

    _OP_MAP: ClassVar[dict[str, str]] = {
        "==": "=",
        "$eq": "=",
        "!=": "!=",
        "$ne": "!=",
        ">": ">",
        "$gt": ">",
        ">=": ">=",
        "$gte": ">=",
        "<": "<",
        "$lt": "<",
        "<=": "<=",
        "$lte": "<=",
    }

    def translate(self, filters: dict[str, Any], params: list[Any]) -> str:
        """
        Translate the given filter dict into an SQL fragment, adding parameters to the params list.

        :param filters: Filter dictionary
        :param params: List to append parameter values to
        :return: SQL WHERE clause fragment
        """
        op = filters.get("operator")

        if op in ("AND", "OR", "NOT", "$and", "$or", "$not"):
            conditions = filters.get("conditions", [])
            if not conditions:
                msg = f"Logical operator {op!r} requires a non-empty 'conditions' list."
                raise ValueError(msg)

            if op in ("NOT", "$not"):
                if len(conditions) != 1:
                    msg = "NOT operator requires exactly one condition."
                    raise ValueError(msg)
                return f"(NOT {self.translate(conditions[0], params)})"

            logical_op = "AND" if op in ("AND", "$and") else "OR"
            translated = [self.translate(cond, params) for cond in conditions]
            return f"({f' {logical_op} '.join(translated)})"

        field = filters.get("field")
        if not field:
            msg = "Comparison filters must include a 'field' key."
            raise ValueError(msg)

        if not op:
            msg = "Each filter condition must include an 'operator' key."
            raise ValueError(msg)

        # Check if 'value' key exists (it's required for comparison operators)
        if "value" not in filters:
            msg = "Comparison filters must include a 'value' key."
            raise ValueError(msg)

        value = filters.get("value")

        if op in ("in", "not in", "$in", "$nin"):
            if not isinstance(value, list) or not value:
                msg = f"Operator {op!r} requires a non-empty list value."
                raise ValueError(msg)

            field_expr = self._field_to_sql(field)
            placeholders = ", ".join("?" for _ in value)
            params.extend(_normalize_value(v) for v in value)

            if op in ("in", "$in"):
                return f"{field_expr} IN ({placeholders})"
            else:
                return f"({field_expr} IS NULL OR {field_expr} NOT IN ({placeholders}))"

        sql_operator = self._OP_MAP.get(op)
        if sql_operator is None:
            msg = f"Unsupported filter operator: {op!r}"
            raise ValueError(msg)

        field_expr = self._field_to_sql(field)

        # Validate value types for comparison operators
        if op in _RANGE_OPS:
            # Range operators (>, >=, <, <=) work with numeric values and ISO date strings
            # Allow None (will be handled by parameterized query), numbers, and ISO date strings
            if value is not None and not isinstance(value, (int, float)) and not _is_iso_date(value):
                msg = f"Operator {op!r} requires a numeric value or ISO date string, got {type(value).__name__}"
                raise ValueError(msg)

        # Handle != operator to include NULL values (documents without the field)
        if sql_operator == "!=":
            params.append(_normalize_value(value))
            return f"({field_expr} IS NULL OR {field_expr} != ?)"

        # For all other operators, use parameterized queries
        params.append(_normalize_value(value))

        return f"{field_expr} {sql_operator} ?"

    @staticmethod
    def _field_to_sql(field: str) -> str:
        """
        Convert a field name to its SQL expression.

        :param field: Field name (e.g., "id", "content", "meta.author")
        :return: SQL expression for the field
        """
        if field in ("id", "content", "embedding"):
            return field

        if field.startswith("meta."):
            field_path = field[5:]
        else:
            field_path = field

        return f"CAST(JSON_VALUE(SYSTOOLS.BSON2JSON(meta), '$.{field_path}') AS VARCHAR(1000))"


def _normalize_value(value: Any) -> Any:
    """
    Normalize a Python value for binding against a JSON_VALUE comparison.

    Booleans are stored in JSON as ``true``/``false`` and read back by ``JSON_VALUE`` as the
    lowercase strings ``'true'``/``'false'``. ibm_db would otherwise bind a Python bool as the
    integer ``1``/``0``, so the comparison would never match. Convert it explicitly.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _is_iso_date(value: Any) -> bool:
    """Return True if *value* is a string that Python recognises as a valid ISO-8601 datetime."""
    if not isinstance(value, str):
        return False
    try:
        # Replace 'Z' suffix with '+00:00' since fromisoformat doesn't support 'Z'
        normalized_value = value.replace("Z", "+00:00") if value.endswith("Z") else value
        datetime.fromisoformat(normalized_value)
        return True
    except ValueError:
        return False


__all__ = ["FilterTranslator"]

# Made with Bob
