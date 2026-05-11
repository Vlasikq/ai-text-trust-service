"""Тесты CSV-парсера: кодировки, разделители, лимиты, edge cases.

Без БД — pure unit-тесты на байтах.
"""

import pytest

from app.services.csv_parser import CsvParseError, parse_csv


def _row(text_len: int, prefix: str = "") -> str:
    """Построить текст ровно нужной длины."""
    base = prefix + "Текст для анализа достаточно длинный. " * 1000
    return base[:text_len]


class TestEncodings:
    def test_utf8_plain(self):
        body = "text\n" + _row(400) + "\n"
        result = parse_csv(body.encode("utf-8"))
        assert len(result.rows) == 1
        assert result.encoding in ("utf-8", "UTF-8", "ascii")

    def test_utf8_bom(self):
        body = "text\n" + _row(400) + "\n"
        # BOM в начале → парсер должен корректно работать.
        result = parse_csv(b"\xef\xbb\xbf" + body.encode("utf-8"))
        assert len(result.rows) == 1
        assert result.encoding == "utf-8-sig"

    def test_windows_1251(self):
        body = "text\n" + _row(400) + "\n"
        result = parse_csv(body.encode("windows-1251"))
        assert len(result.rows) == 1
        # Проверяем что текст декодировался без мусора (есть кириллица).
        assert "Текст" in result.rows[0].text


class TestDelimiters:
    def test_comma(self):
        body = "id,text\n1," + _row(400) + "\n"
        result = parse_csv(body.encode("utf-8"))
        assert result.delimiter == ","
        assert len(result.rows) == 1
        assert result.rows[0].external_id == "1"

    def test_semicolon(self):
        body = "id;text\n1;" + _row(400) + "\n"
        result = parse_csv(body.encode("utf-8"))
        assert result.delimiter == ";"
        assert len(result.rows) == 1
        assert result.rows[0].external_id == "1"

    def test_tab(self):
        body = "id\ttext\n1\t" + _row(400) + "\n"
        result = parse_csv(body.encode("utf-8"))
        assert result.delimiter == "\t"
        assert len(result.rows) == 1


class TestColumns:
    def test_required_text_column(self):
        body = "id,name\n1,foo\n"
        with pytest.raises(CsvParseError, match="text"):
            parse_csv(body.encode("utf-8"))

    def test_optional_id_and_name(self):
        body = "id,name,text\nE-42,Alice," + _row(400) + "\n"
        result = parse_csv(body.encode("utf-8"))
        assert len(result.rows) == 1
        assert result.rows[0].external_id == "E-42"
        assert result.rows[0].name == "Alice"

    def test_extra_columns_ignored(self):
        body = "text,extra,more\n" + _row(400) + ",foo,bar\n"
        result = parse_csv(body.encode("utf-8"))
        assert len(result.rows) == 1

    def test_text_only(self):
        body = "text\n" + _row(400) + "\n"
        result = parse_csv(body.encode("utf-8"))
        assert len(result.rows) == 1
        assert result.rows[0].external_id is None
        assert result.rows[0].name is None

    def test_uppercase_header_normalized(self):
        body = "TEXT,ID\n" + _row(400) + ",x\n"
        result = parse_csv(body.encode("utf-8"))
        assert len(result.rows) == 1
        assert result.rows[0].external_id == "x"


class TestSkippedRows:
    def test_text_too_short(self):
        body = "text\nslishkom korotko\n" + _row(400) + "\n"
        result = parse_csv(body.encode("utf-8"))
        assert len(result.rows) == 1
        assert len(result.skipped) == 1
        assert "text_too_short" in result.skipped[0].reason

    def test_text_too_long(self):
        body = "text\n" + _row(9000) + "\n"
        result = parse_csv(body.encode("utf-8"))
        assert len(result.rows) == 0
        assert len(result.skipped) == 1
        assert "text_too_long" in result.skipped[0].reason

    def test_empty_text(self):
        # Полностью пустые строки между записями csv-модуль молча выкидывает
        # (так задумано в RFC 4180). Поэтому проверяем явный случай:
        # data-row с заполненным id, но пустой text-ячейкой.
        body = "id,text\n7,\n8," + _row(400) + "\n"
        result = parse_csv(body.encode("utf-8"))
        assert len(result.rows) == 1
        assert len(result.skipped) == 1
        assert result.skipped[0].reason == "empty_text"
        assert result.skipped[0].external_id == "7"


class TestLimits:
    def test_too_many_rows_truncated(self):
        rows = "\n".join([_row(400)] * 110)
        body = "text\n" + rows + "\n"
        result = parse_csv(body.encode("utf-8"), max_rows=100)
        assert len(result.rows) == 100
        assert len(result.skipped) == 10
        assert all(s.reason == "too_many_rows" for s in result.skipped)

    def test_file_too_large(self):
        body = ("text\n" + _row(8000) + "\n") * 100
        with pytest.raises(CsvParseError, match="превышает"):
            parse_csv(body.encode("utf-8"), max_size_bytes=10_000)


class TestErrors:
    def test_empty_file(self):
        with pytest.raises(CsvParseError, match="пуст"):
            parse_csv(b"")

    def test_only_whitespace(self):
        with pytest.raises(CsvParseError, match="пуст"):
            parse_csv(b"   \n\n  \n")

    def test_no_header_no_text_column(self):
        # Один data-line без header → DictReader превратит его в header.
        # Если в нём нет 'text' → ошибка валидации.
        body = "1,2,3\n" + _row(400) + "\n"
        with pytest.raises(CsvParseError):
            parse_csv(body.encode("utf-8"))


class TestRowNumbers:
    def test_row_num_excludes_header(self):
        body = "text\n" + _row(400) + "\n" + _row(400) + "\n"
        result = parse_csv(body.encode("utf-8"))
        assert len(result.rows) == 2
        assert result.rows[0].row_num == 2  # row 1 = header
        assert result.rows[1].row_num == 3
