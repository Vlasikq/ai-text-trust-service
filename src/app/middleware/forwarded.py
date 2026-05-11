"""X-Forwarded-For helper.

Один источник правды для извлечения IP клиента из запроса. За Caddy цепочка
выглядит как `<client>, <intermediate>, <caddy>`. Каждый последующий прокси
ДОПИСЫВАЕТ свой адрес в конец, не в начало — значит верить можно только
последнему элементу (поставленному ближайшим к нам доверенным прокси).
Первый элемент клиент задаёт сам и может вписать туда что угодно.
"""

from __future__ import annotations

from fastapi import Request


def client_ip(request: Request) -> str | None:
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else None
