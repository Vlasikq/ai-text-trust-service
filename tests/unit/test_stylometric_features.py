"""Tests for app.features.stylometric — 45 stylometric features."""

import pandas as pd
import pytest

from app.features.stylometric import (
    ALL_FEATURES,
    FEATURE_GROUPS,
    _char_ngrams,
    _count_normalized,
    _entropy,
    _mattr,
    _self_repetition_rate,
    _split_sentences,
    _word_list,
    extract_all,
)

# ── Sample texts ─────────────────────────────────────────────

HUMAN_LIKE = (
    "Вчера ходили с друзьями в кино, посмотрели новый фильм. "
    "Мне понравилось, хотя концовка показалась немного затянутой. "
    "А ты что думаешь? Стоит ли смотреть?"
)

AI_LIKE = (
    "Искусственный интеллект является одним из ключевых направлений "
    "развития современных технологий. Однако следует отметить, что "
    "применение нейронных сетей в различных областях требует тщательного "
    "анализа. Например, в медицине использование ИИ может значительно "
    "ускорить диагностику. Кроме того, существенный прогресс наблюдается "
    "в контексте автоматизации рутинных задач. Ключевой аспект данного "
    "направления является значительный вклад в определённый контекст "
    "существенный для развития. Таким образом, данное направление "
    "является значительным аспектом научно-технического прогресса."
)

MINIMAL_TEXT = "Привет."

EMPTY_TEXT = ""


# ── Helpers ──────────────────────────────────────────────────


class TestSplitSentences:
    def test_basic_split(self):
        assert _split_sentences("Первое. Второе. Третье.") == [
            "Первое", "Второе", "Третье",
        ]

    def test_question_and_exclamation(self):
        result = _split_sentences("Как дела? Отлично! Да.")
        assert len(result) == 3

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_no_punctuation(self):
        result = _split_sentences("Текст без точек")
        assert result == ["Текст без точек"]


class TestWordList:
    def test_cyrillic_words(self):
        words = _word_list("Привет, мир! Как дела?")
        assert words == ["привет", "мир", "как", "дела"]

    def test_mixed_latin_cyrillic(self):
        words = _word_list("Python — это язык")
        assert "python" in words
        assert "это" in words

    def test_empty(self):
        assert _word_list("") == []

    def test_digits_excluded(self):
        words = _word_list("В 2024 году")
        assert "2024" not in words


class TestCharNgrams:
    def test_bigrams(self):
        result = _char_ngrams("abc", 2)
        assert result == ["ab", "bc"]

    def test_trigrams(self):
        result = _char_ngrams("abcd", 3)
        assert result == ["abc", "bcd"]

    def test_text_shorter_than_n(self):
        assert _char_ngrams("a", 2) == []


class TestEntropy:
    def test_uniform_distribution(self):
        from collections import Counter
        c = Counter({"a": 10, "b": 10, "c": 10, "d": 10})
        assert abs(_entropy(c) - 2.0) < 0.001  # log2(4) = 2

    def test_single_symbol(self):
        from collections import Counter
        c = Counter({"a": 100})
        assert _entropy(c) == 0.0

    def test_empty(self):
        from collections import Counter
        assert _entropy(Counter()) == 0.0


class TestMattr:
    def test_short_text_fallback(self):
        words = ["привет", "мир", "как"]
        result = _mattr(words, window=50)
        assert result == pytest.approx(1.0)  # all unique, 3/3

    def test_long_text_range(self):
        words = ["слово"] * 100
        result = _mattr(words, window=50)
        assert result == pytest.approx(1 / 50)  # one unique word

    def test_diverse_text(self):
        words = [f"слово{i}" for i in range(200)]
        result = _mattr(words, window=50)
        assert result == pytest.approx(1.0)  # all unique in every window


class TestSelfRepetition:
    def test_no_repetition(self):
        words = [f"слово{i}" for i in range(20)]
        assert _self_repetition_rate(words, n=3) == 0.0

    def test_full_repetition(self):
        words = ["а", "б", "в"] * 10
        rate = _self_repetition_rate(words, n=3)
        assert rate > 0.5

    def test_short_text(self):
        assert _self_repetition_rate(["а", "б"], n=3) == 0.0


class TestCountNormalized:
    def test_single_word_match(self):
        words = ["однако", "это", "важно", "однако"]
        result = _count_normalized("однако это важно однако", words, "однако")
        assert result == pytest.approx(2 / 4)

    def test_multi_word_match(self):
        text = "кроме того, это важно. кроме того, это нужно."
        words = _word_list(text)
        result = _count_normalized(text, words, "кроме того")
        assert result == pytest.approx(2 / len(words))

    def test_no_match(self):
        words = ["привет", "мир"]
        result = _count_normalized("привет мир", words, "однако")
        assert result == 0.0


# ── extract_all integration ──────────────────────────────────


class TestExtractAll:
    def test_returns_dataframe(self):
        df = extract_all(AI_LIKE)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_all_features_present(self):
        df = extract_all(AI_LIKE)
        for feature in ALL_FEATURES:
            assert feature in df.columns, f"Missing feature: {feature}"

    def test_feature_count(self):
        assert len(ALL_FEATURES) == 45

    def test_feature_groups_cover_all(self):
        from_groups = []
        for features in FEATURE_GROUPS.values():
            from_groups.extend(features)
        assert sorted(from_groups) == sorted(ALL_FEATURES)

    def test_no_nans(self):
        df = extract_all(AI_LIKE)
        assert not df.isna().any().any(), f"NaN in features: {df.isna().sum()}"

    def test_batch_input(self):
        data = pd.DataFrame({"text": [HUMAN_LIKE, AI_LIKE]})
        df = extract_all(data)
        assert len(df) == 2
        assert list(df.index) == [0, 1]

    def test_string_input(self):
        df = extract_all(AI_LIKE)
        assert len(df) == 1

    def test_preserves_index(self):
        data = pd.DataFrame({"text": [HUMAN_LIKE, AI_LIKE]}, index=[10, 20])
        df = extract_all(data)
        assert list(df.index) == [10, 20]


class TestFeatureValues:
    """Sanity checks: feature values are in reasonable ranges."""

    @pytest.fixture
    def feats(self):
        return extract_all(AI_LIKE).iloc[0]

    def test_text_len_positive(self, feats):
        assert feats["text_len"] == len(AI_LIKE)

    def test_word_count_positive(self, feats):
        assert feats["word_count"] > 0

    def test_sentence_count_positive(self, feats):
        assert feats["sentence_count"] > 0

    def test_avg_word_len_range(self, feats):
        assert 2.0 < feats["avg_word_len"] < 15.0

    def test_ttr_range(self, feats):
        assert 0.0 < feats["ttr"] <= 1.0

    def test_comma_density_nonneg(self, feats):
        assert feats["comma_density"] >= 0.0

    def test_cyrillic_ratio_high(self, feats):
        assert feats["cyrillic_ratio"] > 0.3  # mostly Russian text

    def test_digit_ratio_low(self, feats):
        assert feats["digit_ratio"] < 0.1

    def test_entropy_positive(self, feats):
        assert feats["char_bigram_entropy"] > 0.0
        assert feats["char_trigram_entropy"] > 0.0
        assert feats["word_bigram_entropy"] > 0.0

    def test_mattr_range(self, feats):
        assert 0.0 < feats["mattr_50"] <= 1.0

    def test_self_repetition_range(self, feats):
        assert 0.0 <= feats["self_repetition_rate"] <= 1.0

    def test_punct_density_range(self, feats):
        assert 0.0 < feats["punct_density"] < 1.0


class TestAIMarkers:
    """AI-like text should trigger known AI markers."""

    @pytest.fixture
    def feats(self):
        return extract_all(AI_LIKE).iloc[0]

    @pytest.mark.parametrize(
        "word",
        [
            "является",
            "однако",
            "например",
            "ключевой",
            "значительный",
            "существенный",
            "контекст",
            "аспект",
        ],
    )
    def test_fw_marker_detected(self, feats, word):
        assert feats[f"fw_{word}"] > 0.0

    def test_ends_formulaic(self, feats):
        assert feats["ends_formulaic"] == 1


class TestEdgeCases:
    def test_very_short_text(self):
        df = extract_all(MINIMAL_TEXT)
        assert not df.isna().any().any()
        assert df.iloc[0]["word_count"] == 1

    def test_latin_only_text(self):
        df = extract_all("This is an English sentence. Another one here.")
        assert not df.isna().any().any()
        assert df.iloc[0]["cyrillic_ratio"] == 0.0

    def test_numbers_only(self):
        df = extract_all("12345 67890 11111")
        assert not df.isna().any().any()
        # digits excluded from word_list, so word_count = 0 → safe div
        assert df.iloc[0]["word_count"] == 0

    def test_single_long_sentence(self):
        text = "Слово " * 200
        df = extract_all(text.strip())
        assert df.iloc[0]["sentence_count"] >= 1
