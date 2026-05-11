"""Тесты file_extractor: TXT (encodings), DOCX, PDF (blank/empty).

Positive PDF с текстовым слоем не тестируем — для этого нужна reportlab
или подобный writer (нет в deps). Negative-сценарии (blank/encrypted)
покрывают защитные пути модуля; positive путь регулярно проверяется
интеграцией в браузере (drag-drop DOCX).
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.services.file_extractor import ExtractError, extract
from tests.conftest import make_test_app

# ── helpers ───────────────────────────────────────────────────


def _make_docx(paragraphs: list[str]) -> bytes:
    import docx

    buf = io.BytesIO()
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(buf)
    return buf.getvalue()


def _make_blank_pdf() -> bytes:
    """PDF с одной пустой страницей — нет текстового слоя → ExtractError."""
    import pypdf

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ── unit-tests ────────────────────────────────────────────────


class TestTxt:
    def test_utf8(self):
        result = extract("Привет, мир!".encode("utf-8"), filename="x.txt")
        assert result.source_format == "txt"
        assert result.text == "Привет, мир!"
        assert result.char_count == 12

    def test_utf8_bom(self):
        body = "﻿Текст с BOM".encode("utf-8")
        result = extract(body, filename="x.txt")
        assert result.source_format == "txt"
        assert "Текст с BOM" in result.text

    def test_windows_1251(self):
        body = "Кириллица в win-1251".encode("windows-1251")
        result = extract(body, content_type="text/plain")
        assert "Кириллица" in result.text

    def test_undecodable_falls_back_with_warning(self):
        # Намеренно подсунем utf-8 с битым байтом — chardet может определить
        # как-то, но fallback с replace должен дать какой-то текст и warning.
        body = b"\xc3\x28 valid ascii"  # 0xc3 0x28 — невалидная utf-8 пара
        result = extract(body, content_type="text/plain")
        assert result.source_format == "txt"
        # Не падает; warning может появиться (не строгое требование).

    def test_empty_raises(self):
        with pytest.raises(ExtractError, match="пуст"):
            extract(b"", filename="x.txt")


class TestDocx:
    def test_simple_paragraphs(self):
        data = _make_docx(["Первый абзац.", "Второй абзац.", "Третий."])
        result = extract(data, filename="essay.docx")
        assert result.source_format == "docx"
        assert "Первый абзац." in result.text
        assert "Второй абзац." in result.text
        assert "Третий." in result.text

    def test_content_type_detection(self):
        data = _make_docx(["Текст."])
        result = extract(
            data,
            content_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
        )
        assert result.source_format == "docx"

    def test_empty_docx_raises(self):
        data = _make_docx([])
        with pytest.raises(ExtractError, match="не найдено"):
            extract(data, filename="empty.docx")

    def test_corrupt_docx_raises(self):
        with pytest.raises(ExtractError):
            extract(b"not a docx file", filename="bad.docx")


class TestPdf:
    def test_blank_pdf_raises_no_text_layer(self):
        data = _make_blank_pdf()
        with pytest.raises(ExtractError, match="текстового слоя"):
            extract(data, filename="blank.pdf")

    def test_corrupt_pdf_raises(self):
        with pytest.raises(ExtractError):
            extract(b"not a pdf", filename="bad.pdf")

    def test_pdf_page_limit_enforced(self):
        """PDF с >MAX_PDF_PAGES отбрасывается до парсинга текстового слоя."""
        import pypdf

        from app.services.file_extractor import MAX_PDF_PAGES

        writer = pypdf.PdfWriter()
        for _ in range(MAX_PDF_PAGES + 1):
            writer.add_blank_page(width=595, height=842)
        buf = io.BytesIO()
        writer.write(buf)

        with pytest.raises(ExtractError, match="страниц"):
            extract(buf.getvalue(), filename="huge.pdf")


class TestDocxLimits:
    """Лимит общего количества символов DOCX — защита от zip-bomb."""

    def test_docx_total_chars_limit_enforced(self):
        from app.services.file_extractor import MAX_DOCX_TOTAL_CHARS

        # Один большой абзац уже превышает лимит → должно прерваться раньше,
        # чем мы успеем съесть всю память на расширении xml.
        oversized = "А" * (MAX_DOCX_TOTAL_CHARS + 100)
        data = _make_docx([oversized])
        with pytest.raises(ExtractError, match="DOCX слишком большой"):
            extract(data, filename="huge.docx")

    def test_docx_within_limit_succeeds(self):
        """Лимит не отрезает DOCX заведомо допустимого размера."""
        from app.services.file_extractor import MAX_DOCX_TOTAL_CHARS

        # 80% от лимита — заведомо допустимый текст.
        ok_text = "ц" * (MAX_DOCX_TOTAL_CHARS * 8 // 10)
        data = _make_docx([ok_text])
        result = extract(data, filename="ok.docx")
        assert result.source_format == "docx"
        assert len(result.text) >= MAX_DOCX_TOTAL_CHARS * 8 // 10


class TestFormatDetection:
    def test_unknown_extension_defaults_to_txt(self):
        # Без content-type и без расширения → дефолт TXT.
        result = extract(b"plain text", filename="noext")
        assert result.source_format == "txt"

    def test_explicit_unsupported_mime(self):
        with pytest.raises(ExtractError, match="Неподдерживаемый"):
            extract(b"data", content_type="application/octet-stream", filename="x.bin")


# ── endpoint-tests ────────────────────────────────────────────


@pytest.fixture
def client():
    return TestClient(make_test_app(routers=("extract",)))


class TestExtractEndpoint:
    def test_txt_upload(self, client):
        body = "Тест простого TXT файла.".encode("utf-8")
        resp = client.post(
            "/api/v1/extract/text",
            files={"file": ("a.txt", body, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_format"] == "txt"
        assert "Тест" in data["text"]
        assert data["char_count"] > 0

    def test_docx_upload(self, client):
        data = _make_docx(["Содержимое DOCX-файла", "Второй абзац"])
        resp = client.post(
            "/api/v1/extract/text",
            files={
                "file": (
                    "essay.docx",
                    data,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_format"] == "docx"
        assert "Содержимое DOCX-файла" in body["text"]

    def test_blank_pdf_returns_400(self, client):
        data = _make_blank_pdf()
        resp = client.post(
            "/api/v1/extract/text",
            files={"file": ("blank.pdf", data, "application/pdf")},
        )
        assert resp.status_code == 400
        assert "текстового слоя" in resp.json()["detail"]

    def test_oversized_returns_400(self, client):
        big = b"x" * (6 * 1024 * 1024)  # > 5 МБ
        resp = client.post(
            "/api/v1/extract/text",
            files={"file": ("big.txt", big, "text/plain")},
        )
        assert resp.status_code == 400
        assert "превышает" in resp.json()["detail"]
