"""
Расширенный adversarial test set (v2).

Три метода атаки поверх существующих парафразов (v1):
  1. back_translation: Ru → En → Ru через Helsinki-NLP/opus-mt (transformers)
  2. synonym:          замена части вхождений по словарю ru_synonym_replacements.py
  3. homoglyph:        замена 5-10% кириллических символов визуально похожими латинскими

Источник: AI-тексты из data/splits/test_tm.jsonl (label=1).
Результат: data/adversarial_v2/*.jsonl
"""

import argparse
import json
import logging
import random
import re
import sys
import uuid
from pathlib import Path

_DC = Path(__file__).resolve().parent
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUT_DIR = DATA_DIR / "adversarial_v2"

SEED = 42

# ---------------------------------------------------------------------------
# Гомоглифы: кириллица → визуально идентичная латиница
# ---------------------------------------------------------------------------

HOMOGLYPH_MAP = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "А": "A", "В": "B", "Е": "E",
    "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "У": "Y", "Х": "X",
}


def apply_homoglyph(text: str, rate: float = 0.07) -> str:
    """Заменить ~rate кириллических символов гомоглифами."""
    chars = list(text)
    candidates = [i for i, ch in enumerate(chars) if ch in HOMOGLYPH_MAP]
    n_replace = max(1, int(len(candidates) * rate))
    to_replace = random.sample(candidates, min(n_replace, len(candidates)))
    for idx in to_replace:
        chars[idx] = HOMOGLYPH_MAP[chars[idx]]
    return "".join(chars)


# ---------------------------------------------------------------------------
# Back-translation через Helsinki-NLP models
# ---------------------------------------------------------------------------


_MARIAN = None


def _marian_models():
    """Ленивая загрузка Helsinki-NLP Marian (ru↔en)."""
    global _MARIAN
    if _MARIAN is not None:
        return _MARIAN
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ru_en_name = "Helsinki-NLP/opus-mt-ru-en"
    en_ru_name = "Helsinki-NLP/opus-mt-en-ru"
    ru_en_tok = MarianTokenizer.from_pretrained(ru_en_name)
    ru_en_model = MarianMTModel.from_pretrained(ru_en_name).to(device)
    en_ru_tok = MarianTokenizer.from_pretrained(en_ru_name)
    en_ru_model = MarianMTModel.from_pretrained(en_ru_name).to(device)
    ru_en_model.eval()
    en_ru_model.eval()
    _MARIAN = (device, ru_en_tok, ru_en_model, en_ru_tok, en_ru_model)
    return _MARIAN


def _translate_pieces(
    texts: list[str],
    tokenizer,
    model,
    device,
    max_length: int = 256,
) -> list[str]:
    import torch

    if not texts:
        return []
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    with torch.no_grad():
        out_ids = model.generate(**enc, max_length=max_length)
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


def apply_back_translation(text: str) -> str:
    """Ru → En → Ru через MarianMT (Helsinki-NLP/opus-mt-ru-en + en-ru)."""
    device, ru_en_tok, ru_en_model, en_ru_tok, en_ru_model = _marian_models()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    if not parts:
        return text
    # длинные предложения режем по запятым для стабильности длины
    chunks: list[str] = []
    for p in parts:
        if len(p) <= 400:
            chunks.append(p)
        else:
            chunks.extend([c.strip() for c in p.split(",") if c.strip()])
    en_chunks = _translate_pieces(chunks, ru_en_tok, ru_en_model, device)
    ru_chunks = _translate_pieces(en_chunks, en_ru_tok, en_ru_model, device)
    return " ".join(ru_chunks)


# ---------------------------------------------------------------------------
# Синонимная замена (лексический словарь, без эмбеддингов)
# ---------------------------------------------------------------------------


def apply_synonym_replacement(text: str, rate: float = 0.15) -> str:
    from ru_synonym_replacements import apply_synonym_replacement as _go

    return _go(text, rate=rate, seed=None)


# ---------------------------------------------------------------------------
# Общие утилиты
# ---------------------------------------------------------------------------


def load_ai_texts(split_path: Path, max_count: int = 200) -> list[dict]:
    """Загрузить AI-тексты (label=1) из split файла.

    max_count <= 0 --- все AI-тексты в сплите (полный набор для adversarial_v2).
    """
    texts = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["label"] == 1:
                texts.append(rec)
    random.seed(SEED)
    random.shuffle(texts)
    if max_count is None or max_count <= 0:
        return texts
    return texts[:max_count]


def make_record(text: str, attack_type: str, original: dict) -> dict:
    return {
        "text": text,
        "label": 1,
        "domain": original["domain"],
        "source_model": original.get("source_model", "unknown"),
        "source_name": f"adversarial_v2_{attack_type}",
        "doc_id": f"adv2_{attack_type}_{uuid.uuid4().hex[:8]}",
        "meta": {
            "attack_type": attack_type,
            "original_doc_id": original["doc_id"],
            "original_length": len(original["text"]),
            "adversarial_length": len(text),
        },
    }


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved %d records → %s", len(records), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ATTACK_METHODS = {
    "homoglyph": apply_homoglyph,
    "back_translation": apply_back_translation,
    "synonym": apply_synonym_replacement,
}


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial test set v2")
    parser.add_argument("--method", choices=[*ATTACK_METHODS, "all"], default="all")
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Сколько AI-текстов на метод; 0 = все label=1 из сплита",
    )
    parser.add_argument("--split", type=Path, default=SPLITS_DIR / "test_tm.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    ai_texts = load_ai_texts(args.split, args.count)

    methods = list(ATTACK_METHODS.keys()) if args.method == "all" else [args.method]

    for method in methods:
        log.info("=== Method: %s ===", method)
        attack_fn = ATTACK_METHODS[method]
        records = []

        for i, orig in enumerate(ai_texts):
            try:
                adversarial_text = attack_fn(orig["text"])
            except NotImplementedError as e:
                log.warning("Method %s not implemented: %s", method, e)
                break

            records.append(make_record(adversarial_text, method, orig))
            if (i + 1) % 50 == 0:
                log.info("[%d/%d] %s", i + 1, len(ai_texts), method)

        if records:
            save_jsonl(records, args.output_dir / f"adversarial_{method}.jsonl")

    log.info("Done.")


if __name__ == "__main__":
    main()
