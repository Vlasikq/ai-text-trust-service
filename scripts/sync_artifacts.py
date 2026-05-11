"""Sync ML-артефактов из YC Object Storage в локальный artifacts/transformer/.

Запускается на VM перед стартом контейнеров (worker нуждается в трансформере 1.4 GB).

Bucket `aitrust-artifacts` сделан публичным на чтение, поэтому sync работает
без credentials — anonymous boto3-client с UNSIGNED config. Это упрощает деплой:
не нужны статические ключи на VM, не нужен IAM-token из metadata-сервиса
(YC S3 его не принимает напрямую).

Идемпотентен: если файлы уже есть и совпадают по размеру — skip.

Использование:
    python scripts/sync_artifacts.py \
        --bucket aitrust-artifacts \
        --prefix transformer/model \
        --dst /app/artifacts/transformer/model
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config

YC_ENDPOINT = "https://storage.yandexcloud.net"


def _build_s3_client():
    """Anonymous boto3 client. Bucket публичный на чтение."""
    return boto3.client(
        "s3",
        endpoint_url=YC_ENDPOINT,
        config=Config(signature_version=UNSIGNED, region_name="ru-central1"),
    )


def _list_remote(client, bucket: str, prefix: str) -> list[dict]:
    """List все объекты под prefix."""
    paginator = client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append({"key": obj["Key"], "size": obj["Size"]})
    return objects


def _needs_download(local: Path, remote_size: int) -> bool:
    if not local.exists():
        return True
    if local.stat().st_size != remote_size:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True, help="Prefix in bucket, e.g. transformer/model")
    parser.add_argument("--dst", required=True, help="Local destination dir")
    args = parser.parse_args()

    dst_dir = Path(args.dst).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    client = _build_s3_client()
    objects = _list_remote(client, args.bucket, args.prefix)
    if not objects:
        sys.exit(f"ERROR: no objects under s3://{args.bucket}/{args.prefix}/")

    total_mb = sum(o["size"] for o in objects) / 1024 / 1024
    print(f"Found {len(objects)} objects ({total_mb:.1f} MB)")

    t0 = time.perf_counter()
    downloaded = 0
    skipped = 0
    for obj in objects:
        rel = obj["key"][len(args.prefix) + 1 :] if obj["key"].startswith(args.prefix + "/") else obj["key"]
        local = dst_dir / rel
        local.parent.mkdir(parents=True, exist_ok=True)

        if not _needs_download(local, obj["size"]):
            print(f"  [skip] {rel} (already up-to-date)")
            skipped += 1
            continue

        size_mb = obj["size"] / 1024 / 1024
        ts = time.perf_counter()
        client.download_file(args.bucket, obj["key"], str(local))
        elapsed = time.perf_counter() - ts
        rate = size_mb / elapsed if elapsed > 0 else 0
        print(f"  [ok] {rel} ({size_mb:.1f} MB, {elapsed:.1f}s, {rate:.1f} MB/s)")
        downloaded += 1

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {downloaded} downloaded, {skipped} skipped in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
