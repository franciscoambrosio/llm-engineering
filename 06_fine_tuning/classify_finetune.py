#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Email Classification with a Fine-Tuned Classification Head

v1: 60 examples   → saved to ./lora_classifier/
v2: 120 examples  → saved to ./lora_classifier_v2/

Usage:
  python 06_fine_tuning/classify_finetune.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from local_finetune import get_training_data, get_extended_data

LABEL_MAP  = {"complaint": 0, "sales": 1, "support": 2, "general": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

SAVE_V1 = "./lora_classifier"
SAVE_V2 = "./lora_classifier_v2"


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(data, save_path: str, label: str = ""):
    tag = f" ({label})" if label else ""
    print(f"\n{'='*60}")
    print(f"Fine-Tuning: Classification Head{tag}")
    print(f"{'='*60}")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import get_peft_model, LoraConfig, TaskType
        import torch
    except ImportError as e:
        print(f"   ✗ {e}\n   pip install transformers peft torch")
        return None

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=4)
    model.config.pad_token_id = tokenizer.eos_token_id
    total_params = sum(p.numel() for p in model.parameters())

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, lora_alpha=32, lora_dropout=0.05,
        bias="none", target_modules=["c_attn"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Model: gpt2 + LoRA  |  trainable: {trainable:,} ({trainable/total_params*100:.1f}%)")

    random.seed(42)
    random.shuffle(data)
    split = int(len(data) * 0.8)
    train_data, val_data = data[:split], data[split:]

    by_label = {}
    for ex in data:
        by_label.setdefault(ex["label"], 0)
        by_label[ex["label"]] += 1
    dist = "  ".join(f"{k}: {v}" for k, v in sorted(by_label.items()))
    print(f"   Dataset: {len(train_data)} train / {len(val_data)} val  [{dist}]")

    def encode(examples):
        result = []
        for ex in examples:
            t = tokenizer(ex["text"], truncation=True, max_length=128,
                          padding="max_length", return_tensors="pt")
            result.append({
                "input_ids":      t["input_ids"].squeeze(0),
                "attention_mask": t["attention_mask"].squeeze(0),
                "label":          LABEL_MAP[ex["label"]],
            })
        return result

    train_enc = encode(train_data)
    val_enc   = encode(val_data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    device    = torch.device("cpu")
    model.to(device)

    def run_epoch(batches, train_mode=True):
        model.train() if train_mode else model.eval()
        total_loss, correct = 0.0, 0
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for b in batches:
                ids   = b["input_ids"].unsqueeze(0).to(device)
                mask  = b["attention_mask"].unsqueeze(0).to(device)
                label = torch.tensor([b["label"]], dtype=torch.long).to(device)
                out   = model(input_ids=ids, attention_mask=mask, labels=label)
                if train_mode:
                    optimizer.zero_grad()
                    out.loss.backward()
                    optimizer.step()
                total_loss += out.loss.item()
                correct    += int(out.logits.argmax(dim=-1).item() == b["label"])
        n = len(batches)
        return total_loss / n, correct / n

    best_val_loss  = float("inf")
    no_improve     = 0
    patience       = 3
    print(f"\n   {'Epoch':>5}  {'Train loss':>10}  {'Train acc':>9}  {'Val loss':>8}  {'Val acc':>7}")
    print(f"   {'─'*5}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*7}")

    for epoch in range(15):
        tl, ta = run_epoch(train_enc, train_mode=True)
        vl, va = run_epoch(val_enc,   train_mode=False)

        star = "  *" if vl < best_val_loss else ""
        print(f"   {epoch+1:>5}  {tl:>10.3f}  {ta:>9.0%}  {vl:>8.3f}  {va:>7.0%}{star}")

        if vl < best_val_loss:
            best_val_loss = vl
            no_improve    = 0
            model.save_pretrained(save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n   Early stopping at epoch {epoch+1}  (best val loss: {best_val_loss:.3f})")
                break

    print(f"\n   Saved to {save_path}/")
    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    ("Your website has been down for 3 hours and I'm losing business.", "complaint"),
    ("I'd like to purchase 200 licenses. Can we talk pricing?",         "sales"),
    ("How do I reset my API key from the dashboard?",                   "support"),
    ("Do you support SAML-based SSO for enterprise accounts?",          "sales"),
    ("The refund I requested 10 days ago still hasn't appeared.",       "complaint"),
    ("Can I get a list of all the countries you operate in?",           "general"),
    ("Your last update broke the mobile app completely.",               "complaint"),
    ("What payment methods do you accept?",                             "general"),
    ("We need volume pricing for a 1000-seat deployment.",              "sales"),
    ("I can't export my data. The button does nothing.",                "support"),
]


def load_model(save_path: str, tokenizer):
    from transformers import AutoModelForSequenceClassification
    from peft import PeftModel
    base = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=4)
    base.config.pad_token_id = tokenizer.eos_token_id
    m = PeftModel.from_pretrained(base, save_path)
    m.eval()
    return m


def predict(model, tokenizer, text: str) -> str:
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=128, padding="max_length")
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"],
                       attention_mask=inputs["attention_mask"]).logits
    return ID_TO_LABEL[logits.argmax(dim=-1).item()]


def evaluate(tokenizer):
    print(f"\n{'='*60}")
    print("Evaluation: v1 (60 examples) vs v2 (120 examples)")
    print(f"{'='*60}")

    m_v1 = load_model(SAVE_V1, tokenizer)
    m_v2 = load_model(SAVE_V2, tokenizer)

    print(f"\n   {'Email':<48}  Truth       v1          v2")
    print(f"   {'─'*48}  {'─'*10}  {'─'*10}  {'─'*10}")

    v1_correct = v2_correct = 0
    for email, truth in TESTS:
        p1 = predict(m_v1, tokenizer, email)
        p2 = predict(m_v2, tokenizer, email)
        m1 = "✓" if p1 == truth else "✗"
        m2 = "✓" if p2 == truth else "✗"
        v1_correct += p1 == truth
        v2_correct += p2 == truth
        short = email[:46] + ".." if len(email) > 46 else email
        print(f"   {short:<48}  {truth:<10}  {m1} {p1:<8}  {m2} {p2}")

    n = len(TESTS)
    print(f"\n   Accuracy — v1: {v1_correct}/{n} ({v1_correct/n:.0%})   "
          f"v2: {v2_correct}/{n} ({v2_correct/n:.0%})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Train v2 only (v1 checkpoint already exists)
    if not Path(SAVE_V2).exists():
        print("\nTraining v2 on 120 examples...")
        train(get_extended_data(), save_path=SAVE_V2, label="v2 — 120 examples")
    else:
        print(f"\nFound existing v2 checkpoint — skipping training.")

    evaluate(tokenizer)
