"""CSV-парсер для batch-загрузки текстов.

Контракт:
  - Декодирование: chardet → fallback utf-8-sig → utf-8.
  - Разделитель: автодетект через `csv.Sniffer` среди `,;\\t`; если не определилось — `,`.
  - Required column: `text`.
  - Optional columns: `id` (внешний идентификатор клиента), `name` (метка).
  - Лимиты: размер файла, число строк, длина текста — параметризованы.
  - Невалидные строки (короткий/длинный текст, пустой text) НЕ роняют парсинг,
    а попадают в `skipped` с причиной — клиент видит сводку, не теряет файл целиком.

Privacy: парсер живёт в памяти. На уровне batch endpoint текст складывается
в `analysis_jobs.input_text` (зануляется в воркере после обработки) — модуль
сам по себе ничего не пишет в БД.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import chardet


class CsvParseError(ValueError):
    """Файл не CSV или у него нет ожидаемого формата (нет header, нет колонки text)."""


@dataclass
class ParsedRow:
    """Одна валидная строка CSV → запись для будущего analysis_job.

    `external_id` и `name` — то, что прислал клиент в CSV; они уезжают
    обратно в отчёт, чтобы пользователь сматчил результат со своими данными.
    """

    text: str
    external_id: str | None = None
    name: str | None = None
    row_num: int = 0  # 1-based index исходного файла (вкл. header)


@dataclass
class SkippedRow:
    row_num: int
    reason: str
    external_id: str | None = None


@dataclass
class CsvParseResult:
    rows: list[ParsedRow] = field(default_factory=list)
    skipped: list[SkippedRow] = field(default_factory=list)
    encoding: str = "utf-8"
    delimiter: str = ","


def _detect_encoding(data: bytes) -> str:
    """Определить кодировку. UTF-8 BOM имеет приоритет; иначе chardet; fallback utf-8."""
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    detection = chardet.detect(data) or {}
    enc = detection.get("encoding") or "utf-8"
    # chardet иногда отдаёт `ascii` для коротких файлов — utf-8 их тоже декодирует.
    if enc.lower() == "ascii":
        return "utf-8"
    return enc


def _detect_delimiter(sample: str) -> str:
    """csv.Sniffer на первых 4 КБ. Падает на однострочных файлах → дефолт `,`."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        return ","


def parse_csv(
    data: bytes,
    *,
    max_size_bytes: int = 2 * 1024 * 1024,  # 2 MB
    max_rows: int = 100,
    min_text_len: int = 300,
    max_text_len: int = 8000,
) -> CsvParseResult:
    """Распарсить CSV-байты. Бросает `CsvParseError` на структурных проблемах.

    Поведение для содержимого:
      - text < min_text_len → строка в skipped, обработки не будет.
      - text > max_text_len → строка в skipped (явный отказ обрезать — иначе
        пользователь не понимает, какой именно фрагмент пошёл в анализ).
      - валидных строк > max_rows → обрезаются первые max_rows, остальные
        попадают в skipped с причиной 'too_many_rows'.
    """
    if len(data) > max_size_bytes:
        raise CsvParseError(
            f"Файл превышает лимит {max_size_bytes // 1024} КБ (получено {len(data) // 1024} КБ)"
        )
    if not data.strip():
        raise CsvParseError("Файл пуст")

    encoding = _detect_encoding(data)
    try:
        text = data.decode(encoding, errors="strict")
    except UnicodeDecodeError as e:
        raise CsvParseError(
            f"Не удалось декодировать файл как {encoding}: {e}"
        ) from e

    sample = text[:4096]
    delimiter = _detect_delimiter(sample)

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise CsvParseError("Не удалось прочитать заголовок CSV")

    # Нормализуем имена колонок: lowercase + strip + BOM-cleanup.
    fieldnames = [(fn or "").strip().lstrip("﻿").lower() for fn in reader.fieldnames]
    if "text" not in fieldnames:
        raise CsvParseError(
            f"В CSV должна быть колонка 'text'. Найдены: {reader.fieldnames}"
        )

    # Маппинг от нормализованного имени к оригинальному ключу DictReader.
    name_to_orig = dict(zip(fieldnames, reader.fieldnames, strict=True))

    text_key = name_to_orig["text"]
    id_key = name_to_orig.get("id")
    name_col_key = name_to_orig.get("name")

    result = CsvParseResult(encoding=encoding, delimiter=delimiter)

    # row_num=1 — header; первая data-строка row_num=2.
    for offset, raw in enumerate(reader, start=2):
        ext_id = (raw.get(id_key) or "").strip() if id_key else None
        ext_id = ext_id or None
        text_val = (raw.get(text_key) or "").strip()

        if not text_val:
            result.skipped.append(
                SkippedRow(row_num=offset, reason="empty_text", external_id=ext_id)
            )
            continue
        if len(text_val) < min_text_len:
            result.skipped.append(
                SkippedRow(
                    row_num=offset,
                    reason=f"text_too_short({len(text_val)}<{min_text_len})",
                    external_id=ext_id,
                )
            )
            continue
        if len(text_val) > max_text_len:
            result.skipped.append(
                SkippedRow(
                    row_num=offset,
                    reason=f"text_too_long({len(text_val)}>{max_text_len})",
                    external_id=ext_id,
                )
            )
            continue

        if len(result.rows) >= max_rows:
            result.skipped.append(
                SkippedRow(row_num=offset, reason="too_many_rows", external_id=ext_id)
            )
            continue

        name_val = (raw.get(name_col_key) or "").strip() if name_col_key else None
        result.rows.append(
            ParsedRow(
                text=text_val,
                external_id=ext_id,
                name=name_val or None,
                row_num=offset,
            )
        )

    return result
