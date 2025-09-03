"""
ABSA Multilingual (English + Turkish) — Single-File Trainer
-----------------------------------------------------------
• Predict (aspect_category, sentiment) pairs like (SERVICE#GENERAL, negative) and (FOOD#STYLE_OPTIONS, positive).
• Handles English (SemEval-style XML and AspectTerm/AspectCategory XML) and Turkish (provided XML schema).
• Uses BERTTokenizerFast and BertForSequenceClassification in multi-label mode (sigmoid outputs).
• Adds a validation split and searches the best global decision threshold on the validation set.
• Saves: model, tokenizer, label mapping, and best threshold for inference.

Author: Cem Rifki Aydin
Date: 03.09.2025

"""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
import xml.etree.ElementTree as ET

# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_and_tokenizer(lang: str):
    if lang.lower() == "tr":
        model_name = "dbmdz/bert-base-turkish-uncased"
    else:
        model_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return model_name, tokenizer

# ----------------------------
# Data Loading from XML (EN/TR)
# ----------------------------

def parse_xml_sentences(xml_path: str, lang: str = 'en') -> List[Dict[str, Any]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    examples = []

    for sent in root.findall(".//sentence"):
        text_elt = sent.find("text")
        if text_elt is None or text_elt.text is None:
            continue
        text = text_elt.text.strip()
        pairs: List[Tuple[str, str]] = []

        # Turkish or SemEval-style Opinions
        opinions = sent.find("Opinions")
        if opinions is not None:
            for op in opinions.findall("Opinion"):
                category = op.attrib.get("category")
                polarity = op.attrib.get("polarity")
                if category and polarity:
                    pairs.append((category, polarity.lower()))

        # English AspectCategory style
        acats = sent.find("aspectCategories")
        if acats is not None:
            for ac in acats.findall("aspectCategory"):
                category = ac.attrib.get("category")
                polarity = ac.attrib.get("polarity")
                if category and polarity:
                    pairs.append((category.upper(), polarity.lower()))

        examples.append({"text": text, "pairs": pairs})
    return examples


def build_label_space(examples: List[Dict[str, Any]]) -> List[str]:
    label_set = set()
    for ex in examples:
        for cat, pol in ex["pairs"]:
            label_set.add(f"{cat}__{pol}")
    return sorted(label_set)


def encode_labels(examples: List[Dict[str, Any]], all_labels: List[str]) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(all_labels)}
    Y = np.zeros((len(examples), len(all_labels)), dtype=np.float32)
    for r, ex in enumerate(examples):
        for cat, pol in ex["pairs"]:
            key = f"{cat}__{pol}"
            if key in idx:
                Y[r, idx[key]] = 1.0
    return Y

# ----------------------------
# Torch Dataset
# ----------------------------

class ABSADataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: BertTokenizerFast, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# ----------------------------
# Metrics & Threshold Search
# ----------------------------

@dataclass
class ThresholdSearchResult:
    threshold: float
    precision: float
    recall: float
    f1: float


def evaluate_at_threshold(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> ThresholdSearchResult:
    y_pred = (y_scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    return ThresholdSearchResult(threshold, p, r, f1)


def grid_search_threshold(y_true: np.ndarray, y_scores: np.ndarray, grid=None) -> ThresholdSearchResult:
    if grid is None:
        grid = [round(x, 2) for x in np.linspace(0.1, 0.6, 4)]
    best = None
    for t in grid:
        res = evaluate_at_threshold(y_true, y_scores, t)
        if best is None or res.f1 > best.f1:
            best = res
    return best

# ----------------------------
# Training & Inference
# ----------------------------

def train(args):
    set_seed(args.seed)
    examples = parse_xml_sentences(args.dataset, lang=args.lang)
    if len(examples) == 0:
        raise ValueError("No examples found in the provided XML.")

    all_labels = build_label_space(examples)
    if len(all_labels) == 0:
        raise ValueError("No (category, sentiment) pairs found — cannot train.")

    texts = [ex["text"] for ex in examples]
    Y = encode_labels(examples, all_labels)

    tr_idx, val_idx = train_test_split(np.arange(len(texts)), test_size=args.val_ratio, random_state=args.seed, shuffle=True)
    texts_tr = [texts[i] for i in tr_idx]
    texts_val = [texts[i] for i in val_idx]
    Y_tr = Y[tr_idx]
    Y_val = Y[val_idx]

    model_name, tokenizer = get_model_and_tokenizer(args.lang)
    train_ds = ABSADataset(texts_tr, Y_tr, tokenizer, max_length=args.max_length)
    val_ds = ABSADataset(texts_val, Y_val, tokenizer, max_length=args.max_length)

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(all_labels),
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    def collate_fn(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    trainer.train()

    # Threshold search
    model.eval()
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)
    all_scores = []
    with torch.no_grad():
        for batch in val_loader:
            labels = batch.pop("labels")
            outputs = model(**{k: v.to(model.device) for k, v in batch.items()})
            logits = outputs.logits.detach().cpu().numpy()
            scores = 1 / (1 + np.exp(-logits))
            all_scores.append(scores)
    val_scores = np.vstack(all_scores)

    best = grid_search_threshold(Y_val, val_scores)
    print(f"Best threshold on validation: {best.threshold:.2f} | P={best.precision:.4f} R={best.recall:.4f} F1={best.f1:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": all_labels}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump({"best_threshold": best.threshold}, f)

    y_pred = (val_scores >= best.threshold).astype(int)
    print(classification_report(Y_val, y_pred, zero_division=0, target_names=all_labels))

# ----------------------------
# Load & Predict
# ----------------------------

def load_artifacts(model_dir: str):
    with open(os.path.join(model_dir, "labels.json"), "r", encoding="utf-8") as f:
        label_data = json.load(f)
    with open(os.path.join(model_dir, "threshold.json"), "r", encoding="utf-8") as f:
        thres_data = json.load(f)
    labels = label_data["labels"]
    threshold = float(thres_data.get("best_threshold", 0.5))
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return labels, threshold, tokenizer, model

# This function predicts the aspect category and its sentiment based on samples of texts.
def predict(args):
    labels, threshold, tokenizer, model = load_artifacts(args.model_dir)
    texts: List[str] = args.text if isinstance(args.text, list) else [args.text]
    enc = tokenizer(texts, max_length=args.max_length, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in enc.items()})
        logits = outputs.logits.detach().cpu().numpy()
        scores = 1 / (1 + np.exp(-logits))
        preds = (scores >= threshold).astype(int)

    results = []
    for row in preds:
        row_pairs = []
        for i, v in enumerate(row):
            if v == 1:
                lab = labels[i]
                cat, pol = lab.split("__", 1) if "__" in lab else (lab, "unknown")
                row_pairs.append((cat, pol))
        results.append(row_pairs)

    for t, pairs in zip(texts, results):
        print("TEXT:", t)
        if pairs:
            print("PREDICTIONS:")
            for (cat, pol) in pairs:
                print(f"  - ({cat}, {pol})")
        else:
            print("  - No (aspect, sentiment) predicted at current threshold.")

def main(args):

    if args.predict:
        if not args.model_dir or not args.text:
            raise ValueError("--model_dir and --text required for prediction")
        predict(args)
        return

    if not args.dataset:
        raise ValueError("--dataset required for training")
    train(args)

if __name__ == "__main__":
    main()
