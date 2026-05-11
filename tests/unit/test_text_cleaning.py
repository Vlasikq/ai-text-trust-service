"""Тесты для app.preprocessing.text_cleaning — проверяем до запуска pipeline"""

from app.preprocessing.text_cleaning import (
    clean_text,
    normalize_whitespace,
    strip_markdown,
    truncate_by_sentence,
)

# strip_markdown


class TestStripMarkdown:
    """Все тесты strip_markdown предполагают, что \\n уже нормализован в пробел"""

    def test_headings(self):
        assert strip_markdown("## Заголовок текст") == " Заголовок текст"

    def test_bold(self):
        assert strip_markdown("слово **жирный текст** конец") == "слово жирный текст конец"

    def test_bold_single_star(self):
        assert strip_markdown("слово *курсив* конец") == "слово курсив конец"

    def test_italic_underscore(self):
        assert strip_markdown("слово _курсив_ конец") == "слово курсив конец"

    def test_inline_code(self):
        assert strip_markdown("вызов `func()` здесь") == "вызов func() здесь"

    def test_link(self):
        assert strip_markdown("см. [ссылка](http://example.com) далее") == "см. ссылка далее"

    def test_image(self):
        assert strip_markdown("текст ![alt](img.png) продолжение") == "текст  продолжение"

    def test_unordered_list(self):
        assert strip_markdown("пункт - элемент списка").strip() == "пункт элемент списка"

    def test_ordered_list(self):
        assert strip_markdown("пункт 1. первый элемент").strip() == "пункт первый элемент"

    def test_fenced_code_block(self):
        text = "до ```python\nprint('hello')\n``` после"
        # после normalize_ws это стало бы одной строкой — тестируем на ней
        flat = normalize_whitespace(text)
        result = strip_markdown(flat)
        assert "```" not in result
        assert "до" in result
        assert "после" in result

    def test_hr(self):
        result = strip_markdown("текст --- разделитель")
        assert "---" not in result

    def test_residual_double_stars(self):
        """Незакрытый **bold — должен быть убран финальным проходом"""
        result = strip_markdown("текст ** незакрытый bold")
        assert "**" not in result

    def test_residual_double_underscores(self):
        result = strip_markdown("текст __ незакрытый")
        assert "__" not in result

    def test_multiword_bold_after_normalization(self):
        """Многословный bold, который раньше был на разных строках"""
        # после normalize_ws это одна строка
        text = "начало **длинный жирный текст с пробелами** конец"
        result = strip_markdown(text)
        assert "**" not in result
        assert "длинный жирный текст с пробелами" in result

    def test_plain_text_unchanged(self):
        text = "Обычный текст без разметки. Ничего особенного."
        assert strip_markdown(text) == text

    def test_real_world_gemini_pattern(self):
        """Gemini bold с длинным текстом"""
        text = (
            "Производители: **Для них энергетики станут менее доступными. "
            "Конечно, найдутся лазейки.** Реакция."
        )
        result = strip_markdown(text)
        assert "**" not in result
        assert "Производители:" in result
        assert "Реакция." in result


# normalize_whitespace


class TestNormalizeWhitespace:

    def test_newlines_to_space(self):
        assert normalize_whitespace("строка1\nстрока2") == "строка1 строка2"

    def test_double_newlines(self):
        assert normalize_whitespace("абзац1\n\nабзац2") == "абзац1 абзац2"

    def test_mixed_whitespace(self):
        assert normalize_whitespace("a \n\n  b  \n c") == "a b c"

    def test_carriage_return(self):
        assert normalize_whitespace("a\r\nb") == "a b"

    def test_strip(self):
        assert normalize_whitespace("  text  ") == "text"

    def test_no_change(self):
        assert normalize_whitespace("plain text") == "plain text"


# clean_text (combined pipeline)


class TestCleanText:

    def test_full_pipeline(self):
        text = "## Заголовок\n\n**Жирный** текст\n\n- элемент\n1. номер"
        result = clean_text(text)
        assert "##" not in result
        assert "**" not in result
        assert "\n" not in result
        assert "Заголовок" in result
        assert "Жирный" in result

    def test_plain_text(self):
        text = "Обычный текст. Без разметки."
        assert clean_text(text) == text

    def test_no_double_spaces_in_output(self):
        text = "## Заголовок\n\nТекст  с   пробелами\n\n- пункт"
        result = clean_text(text)
        assert "  " not in result

    def test_multiline_bold_real_case(self):
        """Реальный кейс: **bold** растянутый через абзацы."""
        text = "Начало **жирный\nтекст на\nнескольких строках** конец."
        result = clean_text(text)
        assert "**" not in result
        assert "жирный" in result
        assert "конец." in result


# truncate_by_sentence


class TestTruncateBySentence:

    def test_short_text_unchanged(self):
        text = "Короткий текст."
        assert truncate_by_sentence(text, 1000) == text

    def test_truncate_at_period(self):
        text = "Первое предложение. Второе предложение. Третье предложение."
        result = truncate_by_sentence(text, 40)
        assert result.endswith(".")
        assert len(result) <= 40

    def test_truncate_at_question(self):
        text = "Вопрос? Ответ длинный и подробный."
        result = truncate_by_sentence(text, 10)
        assert result == "Вопрос?"

    def test_fallback_to_space(self):
        text = "Оченьдлинноесловобезточек и ещё слова для теста"
        result = truncate_by_sentence(text, 35)
        assert len(result) <= 35
        assert result == result.strip()

    def test_exact_boundary(self):
        text = "A" * 100
        result = truncate_by_sentence(text, 100)
        assert result == text
