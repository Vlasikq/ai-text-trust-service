"""Верификация собранных splits"""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

SPLITS_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"
SPLIT_NAMES = ["train", "val", "test", "holdout_unseen"]
TM_SPLIT_NAMES = ["train_tm", "val_tm", "test_tm", "holdout_tm_unseen"]


@pytest.fixture(scope="module")
def splits():
    result = {}
    for name in SPLIT_NAMES:
        path = SPLITS_DIR / f"{name}.jsonl"
        if path.exists():
            result[name] = pd.read_json(path, lines=True)
    return result


@pytest.fixture(scope="module")
def summary():
    path = SPLITS_DIR / "summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


class TestSplitsExist:
    def test_all_splits_present(self, splits):
        for name in SPLIT_NAMES:
            assert name in splits, f"Split {name} не найден"

    def test_summary_present(self, summary):
        assert summary, "summary.json не найден"

    def test_splits_not_empty(self, splits):
        for name, df in splits.items():
            assert len(df) > 0, f"{name} пустой"


class TestSchema:
    REQUIRED_COLS = {"text", "label", "domain", "source_model", "source_name", "doc_id"}

    def test_columns(self, splits):
        for name, df in splits.items():
            actual = set(df.columns)
            missing = self.REQUIRED_COLS - actual
            assert not missing, f"{name}: нет колонок {missing}"

    def test_label_values(self, splits):
        for name, df in splits.items():
            assert set(df["label"].unique()) <= {0, 1}, f"{name}: неожиданные label"

    def test_no_null_text(self, splits):
        for name, df in splits.items():
            assert df["text"].isna().sum() == 0, f"{name}: есть null в text"

    def test_no_empty_text(self, splits):
        for name, df in splits.items():
            empty = (df["text"].str.strip() == "").sum()
            assert empty == 0, f"{name}: {empty} пустых текстов"


class TestNoMarkdown:
    """Проверка что markdown-артефакты полностью убраны"""

    def test_no_headings(self, splits):
        pat = re.compile(r"(?:^|\s)#{1,3}\s")
        for name, df in splits.items():
            n = df["text"].apply(lambda t: bool(pat.search(t))).sum()
            assert n == 0, f"{name}: {n} текстов с ## headings"

    def test_no_bold(self, splits):
        pat = re.compile(r"\*\*")
        for name, df in splits.items():
            n = df["text"].apply(lambda t: bool(pat.search(t))).sum()
            assert n == 0, f"{name}: {n} текстов с **bold**"

    def test_no_newlines(self, splits):
        for name, df in splits.items():
            n = df["text"].str.contains("\n", regex=False).sum()
            assert n == 0, f"{name}: {n} текстов с \\n"

    def test_no_code_blocks(self, splits):
        pat = re.compile(r"```")
        for name, df in splits.items():
            n = df["text"].apply(lambda t: bool(pat.search(t))).sum()
            assert n == 0, f"{name}: {n} текстов с ```code```"

    def test_no_double_spaces(self, splits):
        for name, df in splits.items():
            n = df["text"].str.contains("  ", regex=False).sum()
            assert n == 0, f"{name}: {n} текстов с двойными пробелами"


class TestNoLeakage:
    """Cross-split leakage check"""

    def test_no_text_overlap_train_val(self, splits):
        train_texts = set(splits["train"]["text"])
        val_texts = set(splits["val"]["text"])
        overlap = train_texts & val_texts
        assert len(overlap) == 0, f"train ↔ val: {len(overlap)} общих текстов"

    def test_no_text_overlap_train_test(self, splits):
        train_texts = set(splits["train"]["text"])
        test_texts = set(splits["test"]["text"])
        overlap = train_texts & test_texts
        assert len(overlap) == 0, f"train ↔ test: {len(overlap)} общих текстов"

    def test_no_text_overlap_train_holdout(self, splits):
        train_texts = set(splits["train"]["text"])
        holdout_texts = set(splits["holdout_unseen"]["text"])
        overlap = train_texts & holdout_texts
        assert len(overlap) == 0, f"train ↔ holdout: {len(overlap)} общих текстов"

    def test_no_text_overlap_val_test(self, splits):
        val_texts = set(splits["val"]["text"])
        test_texts = set(splits["test"]["text"])
        overlap = val_texts & test_texts
        assert len(overlap) == 0, f"val ↔ test: {len(overlap)} общих текстов"


class TestBalance:
    """Проверка баланса классов и доменов"""

    def test_train_class_balance(self, splits):
        vc = splits["train"]["label"].value_counts(normalize=True)
        assert vc.min() > 0.40, f"train: minority class {vc.min():.1%} < 40%"

    def test_val_class_balance(self, splits):
        vc = splits["val"]["label"].value_counts(normalize=True)
        assert vc.min() > 0.40, f"val: minority class {vc.min():.1%} < 40%"

    def test_holdout_is_balanced(self, splits):
        vc = splits["holdout_unseen"]["label"].value_counts()
        assert vc[0] == vc[1], f"holdout: {vc[0]} human != {vc[1]} AI"

    def test_all_domains_in_train(self, splits):
        domains = set(splits["train"]["domain"].unique())
        assert domains == {"essay", "news", "wiki"}, f"train domains: {domains}"

    def test_all_domains_in_holdout(self, splits):
        domains = set(splits["holdout_unseen"]["domain"].unique())
        assert domains == {"essay", "news", "wiki"}, f"holdout domains: {domains}"


class TestHoldout:
    """Проверка holdout split"""

    def test_holdout_only_yandexgpt(self, splits):
        ai_models = splits["holdout_unseen"][
            splits["holdout_unseen"]["label"] == 1
        ]["source_model"].unique()
        assert list(ai_models) == ["yandexgpt"], f"holdout AI models: {ai_models}"

    def test_yandexgpt_not_in_train(self, splits):
        train_models = set(splits["train"]["source_model"].unique())
        assert "yandexgpt" not in train_models, "yandexgpt в train!"

    def test_yandexgpt_not_in_val(self, splits):
        val_models = set(splits["val"]["source_model"].unique())
        assert "yandexgpt" not in val_models, "yandexgpt в val!"


class TestAdversarial:
    """Adversarial тексты — только в test"""

    def test_adversarial_only_in_test(self, splits):
        for name in ["train", "val", "holdout_unseen"]:
            n_adv = (splits[name]["source_model"] == "adversarial").sum()
            assert n_adv == 0, f"{name}: {n_adv} adversarial текстов (должно быть 0)"

    def test_adversarial_in_test(self, splits):
        n_adv = (splits["test"]["source_model"] == "adversarial").sum()
        assert n_adv > 0, "Нет adversarial текстов в test"


class TestTextLength:
    """Проверка фильтрации по длине"""

    def test_min_length(self, splits):
        for name, df in splits.items():
            min_len = df["text"].str.len().min()
            assert min_len >= 300, f"{name}: min length {min_len} < 300"

    def test_max_length(self, splits):
        for name, df in splits.items():
            max_len = df["text"].str.len().max()
            assert max_len <= 8000, f"{name}: max length {max_len} > 8000"


# ── Topic-matched splits (DEC-008) ──────────────────────────────


@pytest.fixture(scope="module")
def tm_splits():
    result = {}
    for name in TM_SPLIT_NAMES:
        path = SPLITS_DIR / f"{name}.jsonl"
        if path.exists():
            result[name] = pd.read_json(path, lines=True)
    return result


class TestTMSplitsExist:
    def test_all_tm_splits_present(self, tm_splits):
        for name in TM_SPLIT_NAMES:
            assert name in tm_splits, f"Split {name} not found"

    def test_tm_splits_not_empty(self, tm_splits):
        for name, df in tm_splits.items():
            assert len(df) > 0, f"{name} is empty"


class TestTMNoLeakage:
    def test_no_docid_overlap_train_val(self, tm_splits):
        overlap = set(tm_splits["train_tm"]["doc_id"]) & set(tm_splits["val_tm"]["doc_id"])
        assert len(overlap) == 0, f"train_tm / val_tm: {len(overlap)} shared doc_ids"

    def test_no_docid_overlap_train_test(self, tm_splits):
        overlap = set(tm_splits["train_tm"]["doc_id"]) & set(tm_splits["test_tm"]["doc_id"])
        assert len(overlap) == 0, f"train_tm / test_tm: {len(overlap)} shared doc_ids"

    def test_no_docid_overlap_val_test(self, tm_splits):
        overlap = set(tm_splits["val_tm"]["doc_id"]) & set(tm_splits["test_tm"]["doc_id"])
        assert len(overlap) == 0, f"val_tm / test_tm: {len(overlap)} shared doc_ids"


class TestTMHoldout:
    def test_holdout_only_qwen3(self, tm_splits):
        ai = tm_splits["holdout_tm_unseen"]
        ai_models = ai[ai["label"] == 1]["source_model"].unique()
        assert list(ai_models) == ["qwen/qwen3-32b"], f"holdout AI: {ai_models}"

    def test_qwen3_not_in_train(self, tm_splits):
        models = set(tm_splits["train_tm"]["source_model"].unique())
        assert "qwen/qwen3-32b" not in models, "qwen3 in train_tm!"

    def test_qwen3_not_in_val(self, tm_splits):
        models = set(tm_splits["val_tm"]["source_model"].unique())
        assert "qwen/qwen3-32b" not in models, "qwen3 in val_tm!"


class TestTMBalance:
    def test_train_class_balance(self, tm_splits):
        vc = tm_splits["train_tm"]["label"].value_counts(normalize=True)
        assert vc.min() > 0.45, f"train_tm: minority {vc.min():.1%} < 45%"

    def test_holdout_is_balanced(self, tm_splits):
        vc = tm_splits["holdout_tm_unseen"]["label"].value_counts()
        assert vc[0] == vc[1], f"holdout_tm: {vc[0]} human != {vc[1]} AI"

    def test_all_domains_in_train(self, tm_splits):
        domains = set(tm_splits["train_tm"]["domain"].unique())
        assert domains == {"essay", "news", "wiki"}, f"train_tm domains: {domains}"


class TestTMAdversarial:
    def test_adversarial_only_in_test(self, tm_splits):
        for name in ["train_tm", "val_tm", "holdout_tm_unseen"]:
            n = (tm_splits[name]["source_model"] == "adversarial").sum()
            assert n == 0, f"{name}: {n} adversarial (should be 0)"

    def test_adversarial_in_test(self, tm_splits):
        n = (tm_splits["test_tm"]["source_model"] == "adversarial").sum()
        assert n > 0, "No adversarial in test_tm"


class TestTMTextLength:
    def test_min_length(self, tm_splits):
        for name, df in tm_splits.items():
            min_len = df["text"].str.len().min()
            assert min_len >= 300, f"{name}: min {min_len} < 300"

    def test_max_length(self, tm_splits):
        for name, df in tm_splits.items():
            max_len = df["text"].str.len().max()
            assert max_len <= 8000, f"{name}: max {max_len} > 8000"
