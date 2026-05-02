#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Fine-Tuning with LoRA

Fine-tune small models locally on your machine.
No cloud API, no cost, runs in minutes.

Requirements:
  pip install transformers peft torch datasets

Usage:
  python local_finetune.py
"""

import json
from pathlib import Path
from typing import List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────

def get_training_data() -> List[Dict[str, str]]:
    """Training data: email classification examples."""
    return [
        # --- complaint ---
        {"text": "My order arrived damaged. The screen is cracked and won't turn on. I'm very disappointed.", "label": "complaint"},
        {"text": "I waited 2 hours for customer service. This is unacceptable!", "label": "complaint"},
        {"text": "The product quality is terrible. I want a refund immediately.", "label": "complaint"},
        {"text": "This is the third time I've had this issue. I'm done with your company.", "label": "complaint"},
        {"text": "I was charged twice for the same order. This is a serious billing error.", "label": "complaint"},
        {"text": "The delivery was 2 weeks late with no communication from your team.", "label": "complaint"},
        {"text": "Your customer service agent was rude and hung up on me.", "label": "complaint"},
        {"text": "The item I received is completely different from what I ordered.", "label": "complaint"},
        {"text": "I've been waiting 3 weeks for my refund and still nothing.", "label": "complaint"},
        {"text": "The packaging was so poor that the product arrived broken.", "label": "complaint"},
        {"text": "My subscription was cancelled without any notice from you.", "label": "complaint"},
        {"text": "The product stopped working after just two days of use.", "label": "complaint"},
        {"text": "I was promised next-day delivery but it arrived 5 days later.", "label": "complaint"},
        {"text": "Your app deleted all my data after the last update. I'm furious.", "label": "complaint"},
        {"text": "I've sent 4 emails and nobody has responded to me yet.", "label": "complaint"},

        # --- sales ---
        {"text": "What's your enterprise pricing? We're interested in buying 100 licenses.", "label": "sales"},
        {"text": "Do you offer bulk discounts? We're a non-profit organization.", "label": "sales"},
        {"text": "Can you send me information about your B2B solutions?", "label": "sales"},
        {"text": "We're evaluating vendors for a 500-seat deployment. Can we schedule a demo?", "label": "sales"},
        {"text": "What's included in the Pro plan vs the Business plan?", "label": "sales"},
        {"text": "Do you have reseller or partner programs available?", "label": "sales"},
        {"text": "I'd like to upgrade our team from the free tier to a paid plan.", "label": "sales"},
        {"text": "Can you provide a custom quote for 250 users?", "label": "sales"},
        {"text": "Are there any annual billing discounts compared to monthly?", "label": "sales"},
        {"text": "We're a startup — do you offer startup pricing or credits?", "label": "sales"},
        {"text": "I want to add 10 more seats to our existing subscription.", "label": "sales"},
        {"text": "Does your enterprise plan include SSO and SAML integration?", "label": "sales"},
        {"text": "We need white-labeling options. Is that available in any plan?", "label": "sales"},
        {"text": "Can I talk to your sales team about a volume purchase?", "label": "sales"},
        {"text": "What's the minimum contract length for enterprise agreements?", "label": "sales"},

        # --- support ---
        {"text": "I'm getting an error 'Connection timeout' when trying to log in. How do I fix this?", "label": "support"},
        {"text": "The app keeps crashing on my iPhone. What should I do?", "label": "support"},
        {"text": "How do I integrate your API with my system?", "label": "support"},
        {"text": "I forgot my password and the reset email isn't arriving. What now?", "label": "support"},
        {"text": "How do I export my data in CSV format?", "label": "support"},
        {"text": "The dashboard isn't loading. I just see a blank white screen.", "label": "support"},
        {"text": "Can you help me set up two-factor authentication on my account?", "label": "support"},
        {"text": "I'm unable to upload files larger than 5MB. Is there a limit?", "label": "support"},
        {"text": "How do I add a new team member to our workspace?", "label": "support"},
        {"text": "My webhook isn't receiving events. How do I debug this?", "label": "support"},
        {"text": "The mobile app won't sync with the desktop version. Help!", "label": "support"},
        {"text": "How do I configure email notifications for my account?", "label": "support"},
        {"text": "I accidentally deleted a file. Is there a way to recover it?", "label": "support"},
        {"text": "The search feature isn't returning results I know exist.", "label": "support"},
        {"text": "How do I transfer ownership of my account to a colleague?", "label": "support"},

        # --- general ---
        {"text": "What are your business hours?", "label": "general"},
        {"text": "Where is your office located?", "label": "general"},
        {"text": "Do you have a phone number I can call?", "label": "general"},
        {"text": "How long has your company been in business?", "label": "general"},
        {"text": "Is your service available in Spanish?", "label": "general"},
        {"text": "Do you have a mobile app for Android?", "label": "general"},
        {"text": "What industries do you typically work with?", "label": "general"},
        {"text": "Are you hiring? I'd love to join your team.", "label": "general"},
        {"text": "Where can I find your press kit?", "label": "general"},
        {"text": "Do you have a community forum or Slack channel?", "label": "general"},
        {"text": "What is your company's privacy policy?", "label": "general"},
        {"text": "Do you comply with GDPR regulations?", "label": "general"},
        {"text": "Is there a free trial available?", "label": "general"},
        {"text": "Where can I read your terms of service?", "label": "general"},
        {"text": "Do you have a blog or newsletter I can subscribe to?", "label": "general"},
    ]


def get_extended_data() -> List[Dict[str, str]]:
    """Original 60 examples + 60 new ones = 120 total (30 per category)."""
    extra = [
        # --- complaint ---
        {"text": "I ordered a laptop but received a keyboard. This is completely wrong.", "label": "complaint"},
        {"text": "My account was charged $200 without any prior notice or explanation.", "label": "complaint"},
        {"text": "The product I received is clearly used, not new as advertised.", "label": "complaint"},
        {"text": "I've called three times this week and no one has resolved my issue.", "label": "complaint"},
        {"text": "Your live chat disconnected me twice without resolving anything.", "label": "complaint"},
        {"text": "The activation code I purchased is invalid. The software won't install.", "label": "complaint"},
        {"text": "I received someone else's order. This is a serious logistics failure.", "label": "complaint"},
        {"text": "My account was suspended without warning or explanation.", "label": "complaint"},
        {"text": "The product description was misleading. It's nothing like the photos.", "label": "complaint"},
        {"text": "I've been locked out of my account for 5 days with no help.", "label": "complaint"},
        {"text": "Your return process is a nightmare. I've been waiting 3 weeks for a label.", "label": "complaint"},
        {"text": "The item broke on first use. This is clearly a manufacturing defect.", "label": "complaint"},
        {"text": "I was overcharged by 50% compared to the price shown on your website.", "label": "complaint"},
        {"text": "Your delivery driver left the package in the rain. Everything is ruined.", "label": "complaint"},
        {"text": "I've asked for a supervisor four times and nobody has called me back.", "label": "complaint"},

        # --- sales ---
        {"text": "We're evaluating your platform for a 300-person engineering team.", "label": "sales"},
        {"text": "Does your pricing change for educational institutions? We're a university.", "label": "sales"},
        {"text": "I'd like to schedule a product demo for our leadership team next week.", "label": "sales"},
        {"text": "Can your product be deployed on-premise, or is it cloud-only?", "label": "sales"},
        {"text": "We're currently using a competitor. What makes your product different?", "label": "sales"},
        {"text": "I want to move from the free plan to Business. How do I upgrade?", "label": "sales"},
        {"text": "Can I try the enterprise features before committing to a contract?", "label": "sales"},
        {"text": "We need a custom contract with specific SLA requirements. Is that possible?", "label": "sales"},
        {"text": "Our procurement team needs a formal vendor questionnaire completed.", "label": "sales"},
        {"text": "Does the Pro plan include priority support or is that an add-on?", "label": "sales"},
        {"text": "We're launching next month and need to onboard 50 users immediately.", "label": "sales"},
        {"text": "Can you provide customer references in the financial services industry?", "label": "sales"},
        {"text": "Is there a minimum number of seats for the enterprise plan?", "label": "sales"},
        {"text": "Our legal team needs a data processing agreement before we proceed.", "label": "sales"},
        {"text": "Do you offer any discounts for annual subscriptions paid upfront?", "label": "sales"},

        # --- support ---
        {"text": "I'm getting a 403 Forbidden error when calling your API. What's wrong?", "label": "support"},
        {"text": "How do I enable dark mode in the desktop application?", "label": "support"},
        {"text": "My team members can't see the documents I shared with them.", "label": "support"},
        {"text": "The billing page is stuck loading. I can't update my payment method.", "label": "support"},
        {"text": "How do I set up a custom domain with your platform?", "label": "support"},
        {"text": "I accidentally added the wrong email to my team. How do I remove it?", "label": "support"},
        {"text": "Can I bulk import users via CSV or do I add them one by one?", "label": "support"},
        {"text": "My notifications stopped working after I updated to the latest version.", "label": "support"},
        {"text": "How do I generate an invoice PDF for our accounting department?", "label": "support"},
        {"text": "I need to revoke API access for a former employee. Where do I do this?", "label": "support"},
        {"text": "How long does it take for new users to receive their verification email?", "label": "support"},
        {"text": "The charts in my dashboard show no data even though I have records.", "label": "support"},
        {"text": "Can I run your app offline or does it require an internet connection?", "label": "support"},
        {"text": "The keyboard shortcuts listed in your docs don't work for me.", "label": "support"},
        {"text": "How do I migrate my data from my old account to a new one?", "label": "support"},

        # --- general ---
        {"text": "Do you offer a student discount for your software?", "label": "general"},
        {"text": "I'd like to leave a review of your product. Where can I do that?", "label": "general"},
        {"text": "What programming languages does your API support?", "label": "general"},
        {"text": "How do I unsubscribe from your marketing emails?", "label": "general"},
        {"text": "Do you have a status page where I can check for outages?", "label": "general"},
        {"text": "Is your platform accessible for users with disabilities?", "label": "general"},
        {"text": "Do you have a referral program for recommending new customers?", "label": "general"},
        {"text": "Where can I find system requirements for the desktop app?", "label": "general"},
        {"text": "Can I use your service from China or are there regional restrictions?", "label": "general"},
        {"text": "Do you support right-to-left languages like Arabic or Hebrew?", "label": "general"},
        {"text": "How often do you release product updates?", "label": "general"},
        {"text": "Is documentation available in languages other than English?", "label": "general"},
        {"text": "What's the best way to reach you in case of an emergency?", "label": "general"},
        {"text": "Do you have a public API documentation page I can share?", "label": "general"},
        {"text": "What is the maximum number of users allowed on a single account?", "label": "general"},
    ]
    return get_training_data() + extra


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

def train_local_model():
    """Fine-tune a model locally with LoRA."""

    print("\n" + "=" * 60)
    print("Local Fine-Tuning with LoRA")
    print("=" * 60)

    # Check imports
    print("\n1. Checking imports...")
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )
        from datasets import Dataset
        from peft import get_peft_model, LoraConfig, TaskType
        import torch

        print("   ✓ All imports successful")
    except ImportError as e:
        print(f"   ✗ Missing package: {e}")
        print("\n   Install with:")
        print("   pip install transformers peft torch datasets")
        return

    # Load model
    print("\n2. Loading model (gpt2)...")
    model_name = "gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # GPT-2 doesn't have a pad token by default, set it to EOS token
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"   ✓ Loaded {model_name} ({sum(p.numel() for p in model.parameters()):,} parameters)")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return

    # Prepare data
    print("\n3. Preparing training data...")
    import random
    data = get_training_data()
    random.seed(42)
    random.shuffle(data)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split:]
    print(f"   ✓ {len(data)} total examples → {len(train_data)} train / {len(val_data)} validation")

    def tokenize(examples):
        result = []
        for ex in examples:
            tokens = tokenizer(
                ex["text"],
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt"
            )
            result.append({"input_ids": tokens["input_ids"].squeeze(0)})
        return result

    # Tokenize
    print("\n4. Tokenizing data...")
    tokenized_train = tokenize(train_data)
    tokenized_val = tokenize(val_data)
    print(f"   ✓ Tokenized {len(tokenized_train)} train, {len(tokenized_val)} val examples")

    # Setup LoRA
    print("\n5. Setting up LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["c_attn"],
    )
    model = get_peft_model(model, peft_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Trainable parameters: {trainable_params:,} (~1% of total)")

    # Simple training loop
    print("\n6. Setting up training...")
    import torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    device = torch.device("cpu")  # Use CPU for compatibility
    model = model.to(device)
    print(f"   ✓ Using device: {device}")

    # Training
    num_epochs = 10
    patience = 3        # stop if val loss doesn't improve for 3 epochs
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []        # (epoch, train_loss, val_loss)

    def loss_bar(loss_val: float, max_loss: float = 15.0, width: int = 20) -> str:
        filled = min(int((loss_val / max_loss) * width), width)
        return "[" + "#" * filled + "." * (width - filled) + "]"

    def eval_loss(batches) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in batches:
                input_ids = batch["input_ids"].unsqueeze(0).to(device)
                total += model(input_ids=input_ids, labels=input_ids).loss.item()
        model.train()
        return total / len(batches) if batches else 0.0

    print(f"\n7. Starting training (up to {num_epochs} epochs, patience={patience})...")

    try:
        for epoch in range(num_epochs):
            total_loss = 0.0
            print(f"\n   --- Epoch {epoch+1}/{num_epochs} ---")
            for i, batch in enumerate(tokenized_train):
                input_ids = batch["input_ids"].unsqueeze(0).to(device)
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                bar = loss_bar(loss.item())
                print(f"   Batch {i+1:2}/{len(tokenized_train)}  Loss: {loss.item():7.4f}  {bar}")

            train_loss = total_loss / len(tokenized_train)
            val_loss = eval_loss(tokenized_val)
            history.append((epoch + 1, train_loss, val_loss))

            improved = "  *best*" if val_loss < best_val_loss else ""
            print(f"\n   Epoch {epoch+1:2}  train: {train_loss:.4f}  val: {val_loss:.4f}{improved}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                model.save_pretrained("./lora_model")  # save best checkpoint
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\n   Early stopping: val loss hasn't improved for {patience} epochs.")
                    print(f"   Best val loss: {best_val_loss:.4f} (epoch {epoch+1 - patience})")
                    break

        print("\n   ✓ Training complete!")
        print(f"\n   Epoch history:")
        print(f"   {'Epoch':>5}  {'Train':>8}  {'Val':>8}  {'Status'}")
        print(f"   {'─'*5}  {'─'*8}  {'─'*8}  {'─'*10}")
        best_epoch = min(history, key=lambda x: x[2])
        for ep, tl, vl in history:
            marker = "<-- best" if ep == best_epoch[0] else ""
            print(f"   {ep:>5}  {tl:>8.4f}  {vl:>8.4f}  {marker}")

    except Exception as e:
        print(f"\n   ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n8. Best model already saved to ./lora_model/ (checkpoint from best val loss)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Model: {model_name}")
    print(f"✓ Train / Val examples: {len(train_data)} / {len(val_data)}")
    print(f"✓ Trainable params: {trainable_params:,}")
    print(f"✓ Best val loss: {best_val_loss:.4f}")
    print(f"✓ Output: ./lora_model/")
    print("\nNext: Load the model and use it for inference")
    print("  from peft import AutoPeftModelForCausalLM")
    print("  model = AutoPeftModelForCausalLM.from_pretrained('./lora_model')")
    print("=" * 60)


if __name__ == "__main__":
    train_local_model()
