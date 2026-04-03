"""Fine-tune ruRoBERTa для детекции AI-текстов"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL = "ai-forever/ruRoBERTa-large"
FALLBACK_MODEL = "ai-forever/ruRoBERTa-base"


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
        "auc_roc": roc_auc_score(labels, probs),
    }


def load_data(data_dir: Path):
    train = pd.read_json(data_dir / "train_tm.jsonl", lines=True)
    val = pd.read_json(data_dir / "val_tm.jsonl", lines=True)

    log.info(f"Train: {len(train)} (human={sum(train['label']==0)}, AI={sum(train['label']==1)})")
    log.info(f"Val: {len(val)} (human={sum(val['label']==0)}, AI={sum(val['label']==1)})")

    return (
        train["text"].tolist(),
        train["label"].tolist(),
        val["text"].tolist(),
        val["label"].tolist(),
    )


def train_model(
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
    epochs: int = 5,
    lr: float = 2e-5,
    max_length: int = 512,
    eval_steps: int = 200,
    patience: int = 2,
):
    log.info(f"Model: {model_name}")
    log.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # загрузка данных
    train_texts, train_labels, val_texts, val_labels = load_data(data_dir)

    # tokenizer + model
    log.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification",
    )

    # datasets
    log.info("Tokenizing...")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length)
    log.info(f"Train tokens shape: {train_dataset.encodings['input_ids'].shape}")
    log.info(f"Val tokens shape: {val_dataset.encodings['input_ids'].shape}")

    # training args
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    # train
    log.info("Starting training...")
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        log.error(
            f"OOM with {model_name}! "
            f"Try --model {FALLBACK_MODEL} or --batch-size 2"
        )
        return None

    # eval on val
    results = trainer.evaluate()
    log.info(f"Val results: {results}")

    # save
    save_dir = output_dir / "model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    log.info(f"Model saved to {save_dir}")

    # save metrics
    metrics = {
        "model_name": model_name,
        "val_f1_macro": results.get("eval_f1_macro"),
        "val_accuracy": results.get("eval_accuracy"),
        "val_auc_roc": results.get("eval_auc_roc"),
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "epochs": epochs,
        "lr": lr,
        "max_length": max_length,
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "best_step": trainer.state.best_model_checkpoint,
    }
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Metrics saved to {metrics_path}")

    return trainer


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ruRoBERTa for AI text detection")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--data-dir", default=None,
        help="Path to splits dir (auto-detected if not set)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Where to save model (default: artifacts/transformer)",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--patience", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    # auto-detect data dir
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif Path("/kaggle/input/ai-text-trust-splits-tm").exists():
        data_dir = Path("/kaggle/input/ai-text-trust-splits-tm")
        log.info("Kaggle detected")
    else:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "splits"

    # output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif Path("/kaggle/working").exists():
        output_dir = Path("/kaggle/working/transformer")
    else:
        output_dir = Path(__file__).resolve().parent.parent / "artifacts" / "transformer"

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Data: {data_dir}")
    log.info(f"Output: {output_dir}")

    if not torch.cuda.is_available():
        log.warning("No GPU detected! Training will be very slow.")
        log.warning("Consider using Kaggle (GPU T4) or adding --model ai-forever/ruRoBERTa-base")

    train_model(
        model_name=args.model,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
        eval_steps=args.eval_steps,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
