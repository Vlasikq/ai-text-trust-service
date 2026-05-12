"""Generate scripts/deploy/docker-compose.prod.yml — то же что render_cloud_init,
но без обёртки cloud-init. Используется при scp-обновлении уже работающей VM.

Usage:
    uv run python scripts/deploy/render_compose_prod.py \\
        --reg-id crpqbj2rjnrhcsfpes38 \\
        --pg-host rc1a-c8tcvredf5pdg33o.mdb.yandexcloud.net \\
        --tag 0.4.0 --web-tag 0.2.0 \\
        --cors-origins ""
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent.parent
ENV_LOCAL = PROJECT_ROOT / ".env.local"


def _read_env(key: str, *, required: bool = True) -> str | None:
    if not ENV_LOCAL.exists():
        if required:
            raise FileNotFoundError(f"{ENV_LOCAL} not found")
        return None
    for line in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    if required:
        raise ValueError(f"{key} not in .env.local")
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reg-id", required=True)
    p.add_argument("--pg-host", required=True)
    p.add_argument("--tag", required=True, help="api image tag")
    p.add_argument("--web-tag", required=True, help="web image tag")
    p.add_argument("--pg-user", default="aitrust")
    p.add_argument("--pg-database", default="aitrust")
    p.add_argument("--cors-origins", default="")
    args = p.parse_args()

    pg_password = _read_env("PG_PASSWORD")
    jwt_secret = _read_env("JWT_SECRET")
    image = f"cr.yandex/{args.reg_id}/aitrust-api:{args.tag}"
    web_image = f"cr.yandex/{args.reg_id}/aitrust-web:{args.web_tag}"
    # quote(safe='') escape'ит спец-символы пароля (:, @, /, ?, #, &, %, пробел).
    db_url = (
        f"postgresql+asyncpg://{args.pg_user}:{quote(pg_password, safe='')}"
        f"@{args.pg_host}:6432/{args.pg_database}?ssl=require"
    )

    compose = (DEPLOY_DIR / "docker-compose.yml").read_text(encoding="utf-8")
    out = (
        compose.replace("__IMAGE__", image)
        .replace("__WEB_IMAGE__", web_image)
        .replace("__DATABASE_URL__", db_url)
        .replace("__JWT_SECRET__", jwt_secret)
        .replace("__CORS_ORIGINS__", args.cors_origins)
    )
    target = DEPLOY_DIR / "docker-compose.prod.yml"
    target.write_text(out, encoding="utf-8")
    print(f"[ok] Wrote {target.relative_to(PROJECT_ROOT)}")
    print(f"  api:  {image}")
    print(f"  web:  {web_image}")


if __name__ == "__main__":
    main()
