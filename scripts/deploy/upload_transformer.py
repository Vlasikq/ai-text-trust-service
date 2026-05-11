"""Загрузка ruRoBERTa-артефактов (config.json, model.safetensors, tokenizer.*)
в YC Object Storage. Запускается локально один раз при подготовке прода.

Авторизация: статический ключ доступа SA `aitrust-sa` из .yc_s3_key.json
(создаётся через `yc iam access-key create --service-account-name aitrust-sa --format json`).

Использование:
    uv run python scripts/deploy/upload_transformer.py \
        --bucket aitrust-artifacts \
        --src artifacts/transformer/model \
        --prefix transformer/model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3
from botocore.client import Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KEY_FILE = PROJECT_ROOT / ".yc_s3_key.json"
YC_ENDPOINT = "https://storage.yandexcloud.net"


def _load_credentials() -> tuple[str, str]:
    if not KEY_FILE.exists():
        sys.exit(
            f"ERROR: {KEY_FILE} not found. Run:\n"
            f"  yc iam access-key create --service-account-name aitrust-sa --format json > .yc_s3_key.json"
        )
    data = json.loads(KEY_FILE.read_text(encoding="utf-8"))
    return data["access_key"]["key_id"], data["secret"]


def _upload_file(client, bucket: str, src: Path, key: str) -> None:
    size_mb = src.stat().st_size / 1024 / 1024
    t0 = time.perf_counter()
    client.upload_file(str(src), bucket, key)
    elapsed = time.perf_counter() - t0
    print(f"  [ok] {key} ({size_mb:.1f} MB, {elapsed:.1f}s, {size_mb / elapsed:.1f} MB/s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="YC Object Storage bucket name")
    parser.add_argument("--src", required=True, help="Local source directory")
    parser.add_argument("--prefix", default="transformer/model", help="Key prefix in bucket")
    args = parser.parse_args()

    src_dir = Path(args.src).resolve()
    if not src_dir.is_dir():
        sys.exit(f"ERROR: {src_dir} is not a directory")

    key_id, secret = _load_credentials()
    client = boto3.client(
        "s3",
        endpoint_url=YC_ENDPOINT,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4", region_name="ru-central1"),
    )

    files = sorted(p for p in src_dir.rglob("*") if p.is_file())
    total_mb = sum(p.stat().st_size for p in files) / 1024 / 1024
    print(f"Uploading {len(files)} files ({total_mb:.1f} MB) -> s3://{args.bucket}/{args.prefix}/")

    t0 = time.perf_counter()
    for path in files:
        rel = path.relative_to(src_dir).as_posix()
        key = f"{args.prefix}/{rel}"
        _upload_file(client, args.bucket, path, key)

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {total_mb:.1f} MB in {elapsed:.1f}s ({total_mb / elapsed:.1f} MB/s)")


if __name__ == "__main__":
    main()
