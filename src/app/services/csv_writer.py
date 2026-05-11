"""CSV-сериализация с защитой от формульных инъекций (OWASP).

Excel/LibreOffice/Numbers интерпретируют ячейку, начинающуюся с `=`, `+`, `-`, `@`,
а также со скрытых управляющих символов, как формулу. При экспорте недоверенных
значений в CSV это превращается в RCE на машине жертвы (DDE, WEBSERVICE и т.п.).

Митигация: префиксуем такие значения одинарной кавычкой — Excel её скрывает,
но больше не интерпретирует содержимое как формулу.
"""

from __future__ import annotations

from typing import Any

_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")


def escape_formula(value: Any) -> Any:
    if isinstance(value, str) and value and value[0] in _FORMULA_TRIGGERS:
        return "'" + value
    return value


def escape_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: escape_formula(v) for k, v in row.items()}
