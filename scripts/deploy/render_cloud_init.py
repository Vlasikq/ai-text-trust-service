"""Render cloud-init.yaml.template + docker-compose.yml + Caddyfile в один cloud-init.

Использование:
    cd <repo>
    PG_HOST=$(yc managed-postgresql cluster get aitrust-pg --format json | jq -r .hosts[0].name)
    REG_ID=$(yc container registry get aitrust-cr --format json | jq -r .id)
    uv run python scripts/deploy/render_cloud_init.py \
        --reg-id "$REG_ID" --pg-host "$PG_HOST" --tag 0.1.0

Читает PG_PASSWORD из .env.local. Пишет результат в scripts/deploy/cloud-init.rendered.yaml
(лежит в .gitignore, потому что содержит DATABASE_URL с паролем).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent.parent
ENV_LOCAL = PROJECT_ROOT / ".env.local"
# Публичные SSH-ключи лежат вне репо: render берёт их отсюда. Файл gitignored,
# не должен попасть в коммит ни при каком сценарии (даже git add -A).
SSH_AUTHORIZED_KEYS_FILE = DEPLOY_DIR / ".local" / "ssh_authorized_keys"


def _read_env(key: str, *, required: bool = True) -> str | None:
    """Прочитать переменную из .env.local. Если required=True — бросает.

    Используется для PG_PASSWORD и JWT_SECRET — оба секрета не должны попасть в git.
    """
    if not ENV_LOCAL.exists():
        if required:
            raise FileNotFoundError(f"{ENV_LOCAL} not found — generate {key} first")
        return None
    for line in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    if required:
        raise ValueError(f"{key} not found in .env.local")
    return None


def _read_pg_password() -> str:
    return _read_env("PG_PASSWORD")


def _read_ssh_authorized_keys() -> str:
    """Собрать YAML-блок ssh_authorized_keys из gitignored файла.

    Файл `scripts/deploy/.local/ssh_authorized_keys` — по одному ключу на строку,
    как в обычном `~/.ssh/authorized_keys`. Пустые строки и комментарии (#) игнорируем.
    Если файла нет — render падает с инструкцией: VM можно поднять и без ключа
    (Serial Console через OS Login), но в дефолте требуем явного шага оператора.
    """
    if not SSH_AUTHORIZED_KEYS_FILE.exists():
        raise FileNotFoundError(
            f"{SSH_AUTHORIZED_KEYS_FILE} не найден. Положите туда публичные SSH-ключи "
            "(по одному на строку, формат authorized_keys). Этот файл в .gitignore.\n"
            "Пример: cp ~/.ssh/aitrust_yc.pub " + str(SSH_AUTHORIZED_KEYS_FILE)
        )
    lines = []
    for raw in SSH_AUTHORIZED_KEYS_FILE.read_text(encoding="utf-8").splitlines():
        key = raw.strip()
        if not key or key.startswith("#"):
            continue
        # 6 пробелов — уровень элемента списка под `ssh_authorized_keys:` в YAML.
        lines.append(f"      - {key}")
    if not lines:
        raise ValueError(f"{SSH_AUTHORIZED_KEYS_FILE} пуст — нужен хотя бы один ключ")
    return "\n".join(lines)


def _indent(text: str, spaces: int = 6) -> str:
    """Indent every line for embedding inside cloud-init `content: |` block."""
    pad = " " * spaces
    return "\n".join(pad + line if line else "" for line in text.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg-id", required=True, help="Container Registry ID")
    parser.add_argument(
        "--pg-host",
        required=True,
        help="Managed PG FQDN (rc1a-xxx.mdb.yandexcloud.net)",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain Caddy выпустит сертификат для (e.g. 89.169.141.35.sslip.io)",
    )
    parser.add_argument("--tag", default="0.1.0", help="Docker image tag (api)")
    parser.add_argument(
        "--web-tag", default="0.1.0", help="Docker image tag for aitrust-web"
    )
    parser.add_argument(
        "--pg-user", default="aitrust", help="PG username (default: aitrust)"
    )
    parser.add_argument(
        "--pg-database", default="aitrust", help="PG database (default: aitrust)"
    )
    parser.add_argument(
        "--cors-origins",
        default="",
        help="Comma-separated CORS origins (e.g. https://aitrust.vercel.app)",
    )
    args = parser.parse_args()

    pg_password = _read_pg_password()
    # JWT_SECRET — обязателен в проде; читаем из .env.local или генерируем ошибку.
    jwt_secret = _read_env("JWT_SECRET")
    image = f"cr.yandex/{args.reg_id}/aitrust-api:{args.tag}"
    web_image = f"cr.yandex/{args.reg_id}/aitrust-web:{args.web_tag}"
    # ssl=require (не verify-full) — для MVP избегаем mount root.crt в контейнер.
    # Если нужен полный TLS verify — добавить volume cert и `?sslrootcert=...&ssl=verify-full`.
    # quote(safe='') escape'ит спец-символы пароля (:, @, /, ?, #, &, %), иначе DSN
    # ломается. Например, пароль `p@ss:w0rd` без quote → `aitrust:p@ss:w0rd@host`.
    db_url = (
        f"postgresql+asyncpg://{args.pg_user}:{quote(pg_password, safe='')}"
        f"@{args.pg_host}:6432/{args.pg_database}?ssl=require"
    )

    compose = (DEPLOY_DIR / "docker-compose.yml").read_text(encoding="utf-8")
    compose = (
        compose.replace("__IMAGE__", image)
        .replace("__WEB_IMAGE__", web_image)
        .replace("__DATABASE_URL__", db_url)
        .replace("__JWT_SECRET__", jwt_secret)
        .replace("__CORS_ORIGINS__", args.cors_origins)
    )

    caddyfile = (DEPLOY_DIR / "Caddyfile").read_text(encoding="utf-8")
    caddyfile = caddyfile.replace("__DOMAIN__", args.domain)
    bootstrap = (DEPLOY_DIR / "bootstrap.sh").read_text(encoding="utf-8")
    bootstrap = bootstrap.replace("__DOMAIN__", args.domain)

    ssh_keys_block = _read_ssh_authorized_keys()

    template = (DEPLOY_DIR / "cloud-init.yaml.template").read_text(encoding="utf-8")
    rendered = template.replace("__COMPOSE_CONTENT__", _indent(compose))
    rendered = rendered.replace("__CADDYFILE_CONTENT__", _indent(caddyfile))
    rendered = rendered.replace("__BOOTSTRAP_CONTENT__", _indent(bootstrap))
    rendered = rendered.replace("__SSH_AUTHORIZED_KEYS__", ssh_keys_block)

    out = DEPLOY_DIR / "cloud-init.rendered.yaml"
    out.write_text(rendered, encoding="utf-8")
    print(f"[ok] Rendered: {out.relative_to(PROJECT_ROOT)}")
    print(f"  Image: {image}")
    print(f"  PG host: {args.pg_host}")
    print(f"  Database URL: postgresql+asyncpg://{args.pg_user}:****@{args.pg_host}:6432/{args.pg_database}?ssl=require")
    print()
    print("Next: yc compute instance create --metadata-from-file user-data=" +
          str(out.relative_to(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
