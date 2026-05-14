"""Тесты app.explanation.markers — StyleExplainer."""

import json
from pathlib import Path

from app.explanation.markers import StyleExplainer
from app.schemas import Explanation

# ── Helpers ───────────────────────────────────────────────────


def _save_baselines(path: Path, baselines: dict) -> Path:
    path.write_text(json.dumps(baselines), encoding="utf-8")
    return path


SAMPLE_TEXT = (
    "Искусственный интеллект является одним из ключевых направлений развития "
    "современных технологий. Однако следует отметить, что применение нейронных "
    "сетей в различных областях требует тщательного анализа и проверки результатов. "
    "Например, в медицине использование ИИ может значительно ускорить диагностику."
)

SAMPLE_BASELINES = {
    "text_len": 300.0,
    "avg_word_len": 5.0,
    "ttr": 0.65,
    "avg_sent_len": 15.0,
    "sentence_len_std": 5.0,
    "punct_density": 0.05,
    "hapax_ratio": 0.4,
    "long_word_ratio": 0.15,
    "self_repetition": 0.02,
    "mattr_50": 0.7,
    "char_bigram_entropy": 4.5,
    "ends_formulaic": 0.0,
    "fw_является": 0.001,
    "fw_однако": 0.001,
    "fw_например": 0.001,
    "fw_ключевой": 0.0005,
}


# ── Tests ─────────────────────────────────────────────────────


class TestStyleExplainer:
    def test_not_ready_without_load(self, tmp_path):
        explainer = StyleExplainer(tmp_path / "baselines.json")
        assert explainer.is_ready() is False

    def test_ready_after_load(self, tmp_path):
        _save_baselines(tmp_path / "baselines.json", SAMPLE_BASELINES)
        explainer = StyleExplainer(tmp_path / "baselines.json")
        explainer.load()
        assert explainer.is_ready() is True

    def test_missing_file_no_crash(self, tmp_path):
        explainer = StyleExplainer(tmp_path / "nonexistent.json")
        explainer.load()  # should not raise
        assert explainer.is_ready() is False

    def test_explain_returns_explanation(self, tmp_path):
        _save_baselines(tmp_path / "baselines.json", SAMPLE_BASELINES)
        explainer = StyleExplainer(tmp_path / "baselines.json")
        explainer.load()

        result = explainer.explain(SAMPLE_TEXT)
        assert isinstance(result, Explanation)
        assert len(result.top_markers) > 0
        assert len(result.top_markers) <= 5

    def test_markers_have_all_fields(self, tmp_path):
        _save_baselines(tmp_path / "baselines.json", SAMPLE_BASELINES)
        explainer = StyleExplainer(tmp_path / "baselines.json")
        explainer.load()

        result = explainer.explain(SAMPLE_TEXT)
        m = result.top_markers[0]
        assert m.feature is not None
        assert isinstance(m.value, float)
        assert isinstance(m.human_baseline, float)
        assert isinstance(m.deviation_percent, float)
        assert m.direction in ("above", "below")

    def test_markers_sorted_by_deviation(self, tmp_path):
        _save_baselines(tmp_path / "baselines.json", SAMPLE_BASELINES)
        explainer = StyleExplainer(tmp_path / "baselines.json")
        explainer.load()

        result = explainer.explain(SAMPLE_TEXT)
        devs = [abs(m.deviation_percent) for m in result.top_markers]
        assert devs == sorted(devs, reverse=True)

    def test_summary_not_empty(self, tmp_path):
        _save_baselines(tmp_path / "baselines.json", SAMPLE_BASELINES)
        explainer = StyleExplainer(tmp_path / "baselines.json")
        explainer.load()

        result = explainer.explain(SAMPLE_TEXT)
        assert len(result.summary) > 0
        assert "отклонения" in result.summary.lower() or "нормы" in result.summary.lower()

    def test_top_n_parameter(self, tmp_path):
        _save_baselines(tmp_path / "baselines.json", SAMPLE_BASELINES)
        explainer = StyleExplainer(tmp_path / "baselines.json")
        explainer.load()

        result = explainer.explain(SAMPLE_TEXT, top_n=3)
        assert len(result.top_markers) <= 3
