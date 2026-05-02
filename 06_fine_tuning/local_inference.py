#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference with Fine-Tuned LoRA Model

Loads the saved adapter and classifies new emails.

Usage:
  python 06_fine_tuning/local_inference.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from local_finetune import get_training_data


def run_inference():
    print("\n" + "=" * 60)
    print("Inference with Fine-Tuned LoRA Model")
    print("=" * 60)

    # Check model exists
    adapter_path = Path("./lora_model")
    if not adapter_path.exists():
        print("\n✗ No saved model found at ./lora_model/")
        print("  Run local_finetune.py first to train and save the model.")
        return

    # Load packages
    print("\n1. Loading packages...")
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        import torch
        print("   ✓ Done")
    except ImportError as e:
        print(f"   ✗ {e}")
        print("   pip install transformers peft torch")
        return

    # Load model + adapter
    print("\n2. Loading base model + LoRA adapter...")
    try:
        model = AutoPeftModelForCausalLM.from_pretrained("./lora_model")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        print("   ✓ Model loaded")
    except Exception as e:
        print(f"   ✗ {e}")
        return

    # Test emails — none of these were in training
    test_emails = [
        "Your website has been down for 3 hours and I'm losing business because of it.",
        "I'd like to purchase 200 licenses for our company. Can we talk pricing?",
        "How do I reset my API key from the dashboard?",
        "Do you support SAML-based single sign-on for enterprise accounts?",
        "The refund I requested 10 days ago still hasn't appeared on my card.",
        "Can I get a list of all the countries you operate in?",
    ]

    # Group all training examples by category
    train_data = get_training_data()
    by_category = {}
    for ex in train_data:
        by_category.setdefault(ex["label"], []).append(ex["text"])

    print(f"\n   Using all training examples per category:")
    for cat, examples in by_category.items():
        print(f"     {cat:<10} {len(examples)} examples")

    def score_text(text: str) -> float:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            loss = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"]).loss
        return loss.item()

    def classify(email: str):
        """Average loss across all training examples in each category."""
        scores = {}
        for cat, examples in by_category.items():
            losses = [score_text(f"{ex} {email}") for ex in examples]
            scores[cat] = sum(losses) / len(losses)
        return min(scores, key=scores.get), scores

    print("\n3. Classifying test emails...\n")
    print(f"   {'Email':<55}  {'Predicted':<10}  Scores")
    print(f"   {'─'*55}  {'─'*10}  {'─'*40}")

    for email in test_emails:
        predicted, scores = classify(email)
        score_str = "  ".join(f"{c[0].upper()}:{s:.2f}" for c, s in scores.items())
        short = email[:52] + "..." if len(email) > 52 else email
        print(f"   {short:<55}  {predicted:<10}  {score_str}")

    # Compare against base model (no fine-tuning)
    print("\n4. Loading base GPT-2 (no fine-tuning)...")
    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    base_model.eval()
    print("   ✓ Base model loaded")

    def score_text_base(text: str) -> float:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            loss = base_model(input_ids=inputs["input_ids"], labels=inputs["input_ids"]).loss
        return loss.item()

    def classify_base(email: str):
        scores = {}
        for cat, examples in by_category.items():
            losses = [score_text_base(f"{ex} {email}") for ex in examples]
            scores[cat] = sum(losses) / len(losses)
        return min(scores, key=scores.get), scores

    print("\n5. Side-by-side comparison...\n")
    print(f"   {'Email':<45}  {'Fine-tuned':<12}  {'Base GPT-2':<12}  Match?")
    print(f"   {'─'*45}  {'─'*12}  {'─'*12}  {'─'*6}")

    ft_correct = 0
    base_correct = 0

    # Ground truth for the 6 test emails
    ground_truth = ["complaint", "sales", "support", "sales", "complaint", "general"]

    for email, truth in zip(test_emails, ground_truth):
        ft_pred, _ = classify(email)
        base_pred, _ = classify_base(email)
        short = email[:42] + "..." if len(email) > 42 else email

        ft_mark = "✓" if ft_pred == truth else "✗"
        base_mark = "✓" if base_pred == truth else "✗"
        match = "same" if ft_pred == base_pred else "diff"

        print(f"   {short:<45}  {ft_mark} {ft_pred:<10}  {base_mark} {base_pred:<10}  {match}")

        ft_correct += ft_pred == truth
        base_correct += base_pred == truth

    print(f"\n   Accuracy — fine-tuned: {ft_correct}/{len(test_emails)}   base: {base_correct}/{len(test_emails)}")


if __name__ == "__main__":
    run_inference()
