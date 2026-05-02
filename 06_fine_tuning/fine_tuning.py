# -*- coding: utf-8 -*-
"""
Module 06 — Fine-Tuning
=======================

Problem: Your LLM system works, but not well enough on your specific task.

Without fine-tuning:
  - Generic model: "Good at everything, great at nothing"
  - Your task: Customer support email classification
  - Result: 70% accuracy (not good enough)

With fine-tuning:
  - Specialized model: Trained on YOUR data
  - Your task: Customer support classification
  - Result: 94% accuracy (production-ready!)

This module covers:
  1. When to fine-tune (vs prompt engineering)
  2. Preparing training data
  3. Fine-tuning process
  4. Evaluating improvement
  5. Deploying fine-tuned model

Why it matters:
  - Task-specific: Model learns YOUR patterns
  - Cost reduction: Smaller model, same performance
  - Speed: Faster inference
  - Privacy: Keep data on-premises

Run:
  python 06_fine_tuning/fine_tuning.py
"""

import json
from typing import List, Dict
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# 1. WHEN TO FINE-TUNE: Decision Framework
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FineTuneDecision:
    """Decision: Should you fine-tune?"""
    task_name: str
    baseline_accuracy: float
    target_accuracy: float
    data_available: int  # Number of training examples
    budget_usd: float  # Monthly budget

    def should_finetune(self) -> tuple[bool, str]:
        """Decide if fine-tuning is worth it."""

        # Too good already
        if self.baseline_accuracy > 0.95:
            return False, "Already >95% accurate, diminishing returns"

        # Not enough data
        if self.data_available < 100:
            return False, "Need at least 100 examples (you have {self.data_available})"

        # Too expensive
        if self.budget_usd < 50:
            return False, "Fine-tuning costs $50-500+ (you have ${self.budget_usd}/month)"

        # Clear gap between baseline and target
        gap = self.target_accuracy - self.baseline_accuracy
        if gap < 0.05:
            return False, "Gap too small (<5%), try prompt engineering first"

        # Worth it!
        return True, f"Good fit: Gap is {gap:.0%}, data is sufficient, budget OK"


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINING DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingExample:
    """One training example (input + expected output)."""
    prompt: str
    completion: str
    category: str  # For organization


class DatasetBuilder:
    """Build fine-tuning dataset."""

    def __init__(self):
        self.examples: List[TrainingExample] = []

    def add_example(self, prompt: str, completion: str, category: str = "general"):
        """Add one training example."""
        self.examples.append(TrainingExample(
            prompt=prompt.strip(),
            completion=completion.strip(),
            category=category
        ))

    def add_examples(self, examples: List[Dict]):
        """Bulk add examples."""
        for ex in examples:
            self.add_example(
                prompt=ex.get("prompt", ""),
                completion=ex.get("completion", ""),
                category=ex.get("category", "general")
            )

    def to_jsonl(self) -> str:
        """Convert to JSONL format for fine-tuning."""
        lines = []
        for ex in self.examples:
            lines.append(json.dumps({
                "messages": [
                    {"role": "user", "content": ex.prompt},
                    {"role": "assistant", "content": ex.completion}
                ]
            }))
        return "\n".join(lines)

    def stats(self) -> Dict:
        """Dataset statistics."""
        return {
            "total_examples": len(self.examples),
            "by_category": self._count_by_category(),
            "avg_prompt_length": sum(len(ex.prompt.split()) for ex in self.examples) / len(self.examples) if self.examples else 0,
            "avg_completion_length": sum(len(ex.completion.split()) for ex in self.examples) / len(self.examples) if self.examples else 0,
        }

    def _count_by_category(self) -> Dict:
        """Count examples by category."""
        counts = {}
        for ex in self.examples:
            counts[ex.category] = counts.get(ex.category, 0) + 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXAMPLE: EMAIL CLASSIFICATION DATASET
# ─────────────────────────────────────────────────────────────────────────────

def create_email_classification_dataset() -> DatasetBuilder:
    """Create training dataset for email classification."""

    builder = DatasetBuilder()

    # Complaint emails
    builder.add_example(
        prompt="My order arrived damaged. The screen is cracked and won't turn on. I'm very disappointed with this purchase.",
        completion="complaint",
        category="complaint"
    )

    builder.add_example(
        prompt="I waited 2 hours for customer service. This is unacceptable!",
        completion="complaint",
        category="complaint"
    )

    builder.add_example(
        prompt="The product quality is terrible. I want a refund immediately.",
        completion="complaint",
        category="complaint"
    )

    # Sales inquiries
    builder.add_example(
        prompt="What's your enterprise pricing? We're interested in buying 100 licenses.",
        completion="sales_inquiry",
        category="sales"
    )

    builder.add_example(
        prompt="Do you offer bulk discounts? We're a non-profit organization.",
        completion="sales_inquiry",
        category="sales"
    )

    builder.add_example(
        prompt="Can you send me information about your B2B solutions?",
        completion="sales_inquiry",
        category="sales"
    )

    # Technical support
    builder.add_example(
        prompt="I'm getting an error 'Connection timeout' when trying to log in. How do I fix this?",
        completion="technical_support",
        category="support"
    )

    builder.add_example(
        prompt="The app keeps crashing on my iPhone. What should I do?",
        completion="technical_support",
        category="support"
    )

    builder.add_example(
        prompt="How do I integrate your API with my system?",
        completion="technical_support",
        category="support"
    )

    # General inquiries
    builder.add_example(
        prompt="What are your business hours?",
        completion="general_inquiry",
        category="general"
    )

    builder.add_example(
        prompt="Where is your office located?",
        completion="general_inquiry",
        category="general"
    )

    return builder


# ─────────────────────────────────────────────────────────────────────────────
# 4. FINE-TUNING WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

class FineTuningWorkflow:
    """Fine-tuning workflow: prepare → upload → train → evaluate."""

    def __init__(self, dataset: DatasetBuilder, task_name: str = "custom_task"):
        self.dataset = dataset
        self.task_name = task_name
        self.training_file_id = None
        self.model_id = None

    def step1_prepare_data(self):
        """Step 1: Prepare and validate training data."""
        print("\n── Step 1: Prepare Data ──────────────────────────")
        stats = self.dataset.stats()

        print(f"Total examples: {stats['total_examples']}")
        print(f"Examples by category:")
        for cat, count in stats['by_category'].items():
            print(f"  - {cat}: {count}")
        print(f"Avg prompt length: {stats['avg_prompt_length']:.0f} words")
        print(f"Avg completion length: {stats['avg_completion_length']:.0f} words")

        # Validation
        issues = []
        if stats['total_examples'] < 10:
            issues.append("⚠️  Too few examples (<10)")
        if any(count < 2 for count in stats['by_category'].values()):
            issues.append("⚠️  Some categories have <2 examples")

        if issues:
            print("\nIssues found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✓ Dataset looks good!")

        return not bool(issues)

    def step2_estimate_cost(self):
        """Step 2: Estimate fine-tuning cost."""
        print("\n── Step 2: Estimate Cost ─────────────────────────")

        # Rough cost estimates (as of 2024)
        num_examples = self.dataset.stats()['total_examples']

        # Claude pricing: ~$0.03 per 1K input tokens
        avg_prompt_tokens = self.dataset.stats()['avg_prompt_length'] * 1.3  # ~1.3 tokens per word
        input_cost = (num_examples * avg_prompt_tokens / 1000) * 0.03

        # Training cost (varies by provider)
        # Anthropic: $0-1 per example for fine-tuning (simplified)
        training_cost_estimate = num_examples * 0.50  # $0.50 per example

        total_cost = input_cost + training_cost_estimate

        print(f"Input preparation: ${input_cost:.2f}")
        print(f"Training cost (estimate): ${training_cost_estimate:.2f}")
        print(f"Total estimate: ${total_cost:.2f}")
        print("\nNote: This is approximate. Check your provider's pricing.")

    def step3_upload_and_train(self):
        """Step 3: Upload data and start training (simulated)."""
        print("\n── Step 3: Upload & Train ────────────────────────")

        # In real flow:
        # 1. Save dataset to file
        # 2. Upload to provider (Anthropic, OpenAI, etc.)
        # 3. Start fine-tuning job
        # 4. Monitor progress

        print("In production, you would:")
        print("  1. Save: dataset.to_jsonl() → training_data.jsonl")
        print("  2. Upload: client.files.create(file=open('training_data.jsonl'))")
        print("  3. Train: client.fine_tuning.jobs.create(training_file_id='...')")
        print("  4. Monitor: Check status until complete (can take hours)")
        print("\nSimulated: Training started...")
        print("✓ Job ID: ft_abc123def456")
        self.model_id = "ft_abc123def456"

    def step4_evaluate(self):
        """Step 4: Evaluate fine-tuned model."""
        print("\n── Step 4: Evaluate ──────────────────────────────")

        print("Baseline (generic model): 70% accuracy")
        print("Fine-tuned model: 94% accuracy")
        print("Improvement: +24%")
        print("\nBreakdown:")
        print("  - Complaints: 98% (was 85%)")
        print("  - Sales: 92% (was 68%)")
        print("  - Support: 91% (was 65%)")
        print("  - General: 89% (was 72%)")

    def step5_deploy(self):
        """Step 5: Deploy fine-tuned model."""
        print("\n── Step 5: Deploy ────────────────────────────────")

        print(f"Model ready: {self.model_id}")
        print("\nDeployment checklist:")
        print("  ✓ Meets accuracy target (94% > 90%)")
        print("  ✓ No regressions on other tasks")
        print("  ✓ Latency acceptable (<1s)")
        print("  ✓ Cost-effective (saves $X/month)")
        print("\nNext: Update production API to use fine-tuned model")

    def run_full_workflow(self):
        """Run complete fine-tuning workflow."""
        print("=" * 60)
        print(f"Fine-Tuning Workflow: {self.task_name}")
        print("=" * 60)

        if not self.step1_prepare_data():
            print("\n❌ Data validation failed. Fix issues and retry.")
            return

        self.step2_estimate_cost()
        self.step3_upload_and_train()
        self.step4_evaluate()
        self.step5_deploy()


# ─────────────────────────────────────────────────────────────────────────────
# 5. LOCAL FINE-TUNING WITH LORA (Efficient, runs on laptop)
# ─────────────────────────────────────────────────────────────────────────────

class LocalFineTuning:
    """Fine-tune small models locally using LoRA (Low-Rank Adaptation)."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name

    def setup_instructions(self):
        """Show how to set up local fine-tuning."""
        instructions = """
── Local Fine-Tuning Setup ───────────────────────────

Step 1: Install required packages
  pip install transformers peft torch datasets

Step 2: Prepare your data
  - Format: List[{"text": "your training text"}]
  - Min: 10 examples
  - Max: Limited by RAM (100K examples on 16GB)

Step 3: Fine-tune with LoRA
  - LoRA = Low-Rank Adaptation
  - Trains only ~1-2% of model parameters
  - Fast: Minutes instead of hours
  - Efficient: Works on CPU or small GPU

Step 4: Use the fine-tuned model
  - Load base model + LoRA weights
  - Deploy: No additional hardware needed

Why LoRA?
  ✓ 90% less disk space (100MB vs 5GB)
  ✓ 90% less memory usage
  ✓ 10x faster training
  ✓ Same quality as full fine-tuning
"""
        return instructions

    def example_code(self):
        """Show example fine-tuning code."""
        code = '''
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# 1. Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Setup LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # Low-rank dimension
    lora_alpha=32,          # Scaling
    lora_dropout=0.05,      # Dropout
    bias="none",
    target_modules=["c_attn"]  # Which layers to train
)
model = get_peft_model(model, peft_config)

# 3. Prepare training data
training_args = TrainingArguments(
    output_dir="./lora_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # Your data
    data_collator=data_collator,
)

# 4. Train (takes minutes on CPU)
trainer.train()

# 5. Save LoRA weights (small files)
model.save_pretrained("./my_lora_model")

# 6. Use the model later
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    "./my_lora_model",
    device_map="auto"
)
'''
        return code

    def demo_local_finetuning(self):
        """Demonstrate local fine-tuning concept."""
        print(self.setup_instructions())
        print("\n── Example Code ──────────────────────────────────")
        print(self.example_code())


# ─────────────────────────────────────────────────────────────────────────────
# DEMOS
# ─────────────────────────────────────────────────────────────────────────────

def demo_decision_framework():
    """Show when to fine-tune."""
    print("\n── Decision Framework: Should You Fine-Tune? ───────")

    scenarios = [
        FineTuneDecision(
            task_name="Email Classification",
            baseline_accuracy=0.70,
            target_accuracy=0.95,
            data_available=500,
            budget_usd=200
        ),
        FineTuneDecision(
            task_name="General Q&A",
            baseline_accuracy=0.98,
            target_accuracy=0.99,
            data_available=1000,
            budget_usd=500
        ),
        FineTuneDecision(
            task_name="Rare Task",
            baseline_accuracy=0.50,
            target_accuracy=0.90,
            data_available=50,  # Too little data
            budget_usd=100
        ),
    ]

    for scenario in scenarios:
        should_finetune, reason = scenario.should_finetune()
        status = "✓ YES" if should_finetune else "✗ NO"
        print(f"\n{scenario.task_name}: {status}")
        print(f"  {reason}")


def demo_dataset_preparation():
    """Show how to prepare training data."""
    print("\n── Dataset Preparation ──────────────────────────────")

    dataset = create_email_classification_dataset()
    stats = dataset.stats()

    print(f"Total examples: {stats['total_examples']}")
    print(f"By category: {stats['by_category']}")
    print(f"Avg prompt: {stats['avg_prompt_length']:.0f} words")
    print(f"Avg completion: {stats['avg_completion_length']:.0f} words")

    print("\nSample JSONL format:")
    jsonl = dataset.to_jsonl()
    print(jsonl.split('\n')[0][:100] + "...")


def demo_full_workflow():
    """Show complete fine-tuning workflow."""
    dataset = create_email_classification_dataset()
    workflow = FineTuningWorkflow(dataset, task_name="Email Classification")
    workflow.run_full_workflow()


def demo_local_finetuning():
    """Show how to fine-tune models locally."""
    local_ft = LocalFineTuning(model_name="gpt2")
    local_ft.demo_local_finetuning()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "local":
        # Run local fine-tuning demo
        demo_local_finetuning()
    else:
        # Run cloud fine-tuning demos
        print("=" * 60)
        print("Module 06: Fine-Tuning")
        print("=" * 60)

        demo_decision_framework()
        demo_dataset_preparation()
        demo_full_workflow()

        print("\n" + "=" * 60)
        print("Want to try LOCAL fine-tuning instead?")
        print("=" * 60)
        print("\nRun: python fine_tuning.py local")
        print("\nThis shows how to fine-tune small models locally:")
        print("  - GPT-2, DistilBERT, etc.")
        print("  - Uses LoRA for efficiency")
        print("  - Runs on laptop (no cloud needed)")
