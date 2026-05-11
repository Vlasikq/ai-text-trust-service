"""Export OpenAPI schema.

  → artifacts/openapi.json  — для CI и генерации клиентов
  → openapi/openapi.yaml    — для просмотра в редакторе

Lifespan не запускается: модели не грузятся, БД не пингуется.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from app.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = PROJECT_ROOT / "artifacts" / "openapi.json"
YAML_PATH = PROJECT_ROOT / "openapi" / "openapi.yaml"


def main() -> None:
    spec = app.openapi()

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2, sort_keys=False)
    print(f"Exported JSON: {JSON_PATH.relative_to(PROJECT_ROOT)}")

    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    with YAML_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)
    print(f"Exported YAML: {YAML_PATH.relative_to(PROJECT_ROOT)}")

    paths = list(spec.get("paths", {}).keys())
    print(f"Routes: {len(paths)} ({', '.join(sorted(paths))})")


if __name__ == "__main__":
    main()
