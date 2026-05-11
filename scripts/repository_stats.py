"""Подсчёт тестов (pytest --collect-only) и строк в src/app. Пишет artifacts/repository_stats.json."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "artifacts" / "repository_stats.json"


def count_tests() -> int:
    env = {**os.environ, "PYTHONPATH": "src"}
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    if r.returncode != 0:
        return -1
    # last line often "N tests collected"
    for line in reversed(r.stdout.splitlines()):
        if "collected" in line:
            parts = line.replace(",", "").split()
            for i, p in enumerate(parts):
                if p == "tests" and i > 0 and parts[i - 1].isdigit():
                    return int(parts[i - 1])
    return -1


def count_app_lines() -> dict:
    root = PROJECT_ROOT / "src" / "app"
    total = 0
    files = 0
    for p in root.rglob("*.py"):
        n = sum(1 for _ in p.open(encoding="utf-8"))
        total += n
        files += 1
    return {"python_files": files, "total_lines": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="print JSON to stdout")
    args = parser.parse_args()
    data = {
        "pytest_tests_collected": count_tests(),
        "src_app": count_app_lines(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.print:
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
