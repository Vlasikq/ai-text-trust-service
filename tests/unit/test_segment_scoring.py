"""Юнит-тесты segment_scoring service + endpoint /api/v1/analyze/segments.

Без БД (segments не сохраняются).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.services.segment_scoring import (
    make_windows,
    score_segments,
    split_sentences,
)
from tests.conftest import FakeDetector, make_test_app

# ── unit-tests на сервис ──────────────────────────────────────


class TestSplitSentences:
    def test_simple(self):
        sents = split_sentences("Первое предложение. Второе предложение. Третье.")
        assert len(sents) == 3
        assert sents[0].text == "Первое предложение."
        assert sents[1].text == "Второе предложение."

    def test_positions_match_source(self):
        text = "Первое. Второе. Третье."
        sents = split_sentences(text)
        # Позиции должны попадать на исходные предложения.
        for s in sents:
            assert text[s.start:s.end] == s.text

    def test_with_questions_and_exclamations(self):
        text = "Это вопрос? Это утверждение. Это восклицание!"
        sents = split_sentences(text)
        assert len(sents) == 3

    def test_empty_text(self):
        assert split_sentences("") == []
        assert split_sentences("   \n\t") == []


class TestMakeWindows:
    def test_default_step_is_2(self):
        # 10 предложений, window=3, step=2 → окна (0,3), (2,5), (4,7), (6,9) +
        # дотягивающее (7, 10) → 5 окон.
        from app.services.segment_scoring import Sentence

        sents = [Sentence(text=f"S{i}", start=i * 5, end=i * 5 + 4) for i in range(10)]
        wins = make_windows(sents, window_size=3, step=2)
        assert wins[0] == (0, 3)
        assert wins[1] == (2, 5)
        # Последнее окно дотягивает до конца:
        assert wins[-1][1] == 10

    def test_short_text_single_window(self):
        from app.services.segment_scoring import Sentence

        sents = [Sentence(text="A", start=0, end=1), Sentence(text="B", start=2, end=3)]
        # window_size=3, sentences=2 → одно окно на весь текст.
        assert make_windows(sents, window_size=3, step=2) == [(0, 2)]

    def test_no_sentences(self):
        assert make_windows([], window_size=3, step=2) == []


class TestScoreSegments:
    def test_predict_called_for_each_window(self):
        text = "Первое предложение. Второе. Третье. Четвёртое. Пятое."
        calls: list[str] = []

        def predict(t: str) -> float:
            calls.append(t)
            return 0.5

        scores = score_segments(text, predict, window_size=2, step=1)
        assert len(scores) == len(calls) >= 2
        # Char-ranges соответствуют исходному тексту.
        for s in scores:
            assert text[s.start_char:s.end_char] == s.text


# ── endpoint-tests ────────────────────────────────────────────


# Для transition_detected используем чередующиеся вероятности 0.9 / 0.1
# через prob_seq в conftest.FakeDetector.


@pytest.fixture
def client():
    return TestClient(
        make_test_app(detector=FakeDetector(prob_seq=[0.9, 0.1]))
    )


class TestSegmentsEndpoint:
    def test_happy_path(self, client):
        text = (
            "Первое предложение текста. Второе предложение для теста. "
            "Третье предложение здесь. Четвёртое идёт следом. "
            "Пятое замыкает абзац. Шестое тоже короткое."
        )
        resp = client.post(
            "/api/v1/analyze/segments",
            json={"text": text, "window_size": 3, "step": 2},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["text_length"] == len(text)
        assert data["window_size"] == 3
        assert data["step"] == 2
        assert len(data["segments"]) >= 2
        # Каждый сегмент имеет валидные char-ranges.
        for seg in data["segments"]:
            assert 0 <= seg["start_char"] < seg["end_char"] <= len(text)
            assert 0 <= seg["prob_ai"] <= 1
        # Чередующийся FakeDetector → transition должен быть detected.
        assert data["summary"]["transition_detected"] is True

    def test_short_text_one_segment(self, client):
        resp = client.post(
            "/api/v1/analyze/segments",
            json={"text": "Одно предложение для теста.", "window_size": 3, "step": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["segments"]) == 1
        assert data["summary"]["n_segments"] == 1

    def test_invalid_window_size_rejected(self, client):
        resp = client.post(
            "/api/v1/analyze/segments",
            json={"text": "abc", "window_size": 0, "step": 2},
        )
        assert resp.status_code == 422

    def test_empty_text_rejected(self, client):
        resp = client.post(
            "/api/v1/analyze/segments",
            json={"text": "", "window_size": 3, "step": 2},
        )
        assert resp.status_code == 422
