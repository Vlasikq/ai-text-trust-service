"""Сохранить размер Docker-образа api в artifacts/docker_image_info.json."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "docker_image_info.json"
COMPOSE = ROOT / "docker" / "docker-compose.yml"


def main() -> int:
    try:
        r = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.Size}}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        print("docker not found", file=sys.stderr)
        return 1
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode

    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    data = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "compose_file": str(COMPOSE.relative_to(ROOT)),
        "docker_images_lines": lines[:40],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
