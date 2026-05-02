# -*- coding: utf-8 -*-
"""
Module 03 — Structured Outputs
==============================

Problem: LLMs generate text, but you often need structured data (JSON, tables, etc).

Without structure:
  Input:  "Extract the person and company from this email"
  Output: "The person is John and the company is Acme Corp"
  Problem: How do you parse this reliably? Brittle regex? NLP?

With structure:
  Input:  "Extract the person and company from this email"
  Output: {"person": "John", "company": "Acme Corp"}
  Benefit: Guaranteed JSON, type-safe, easy to use in code

This module covers:
  1. Pydantic models: define schema, get JSON, validate
  2. Prompting: tell LLM exactly what you want
  3. Retry logic: handle LLM failures gracefully
  4. Validation: catch bad data before using it
  5. Monitoring: track success rates

Why it matters:
- Reliability: catch malformed output before using it
- Type safety: know what fields exist
- Composability: pipe structured output into other systems
- Validation: ensure values make sense (email format, ranges, etc)

Run:
  python 03_structured_outputs/structured.py
"""

import os
import json
import time
from typing import Optional
from collections import defaultdict
from pydantic import BaseModel, Field, field_validator, ValidationError


# ─────────────────────────────────────────────────────────────────────────────
# 1. PYDANTIC MODELS: Define the structure
# ─────────────────────────────────────────────────────────────────────────────

class Contact(BaseModel):
    """Extracted contact information."""
    name: str = Field(..., description="Full name of the person")
    email: str = Field(..., description="Email address")
    company: Optional[str] = Field(None, description="Company name if mentioned")
    role: Optional[str] = Field(None, description="Job title if mentioned")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v or "." not in v.split("@")[1]:
            raise ValueError("Invalid email format")
        return v


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    text: str = Field(..., description="The analyzed text")
    sentiment: str = Field(..., description="Sentiment label: positive, negative, neutral")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    explanation: str = Field(..., description="Brief explanation")

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v):
        if v not in ["positive", "negative", "neutral"]:
            raise ValueError("Sentiment must be positive, negative, or neutral")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class ExtractionResult(BaseModel):
    """Extraction result with validation."""
    text: str = Field(..., description="Input text")
    extracted: list[Contact] = Field(..., description="Extracted contacts")
    count: int = Field(..., description="Number of contacts extracted")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MONITORING: Track success rates
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionMetrics:
    def __init__(self):
        self.tasks = defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "validation_errors": 0,
            "json_errors": 0,
            "other_errors": 0
        })

    def record_attempt(self, task_name: str):
        self.tasks[task_name]["total"] += 1

    def record_success(self, task_name: str):
        self.tasks[task_name]["success"] += 1

    def record_validation_error(self, task_name: str):
        self.tasks[task_name]["validation_errors"] += 1

    def record_json_error(self, task_name: str):
        self.tasks[task_name]["json_errors"] += 1

    def record_other_error(self, task_name: str):
        self.tasks[task_name]["other_errors"] += 1

    def get_success_rate(self, task_name: str) -> float:
        task = self.tasks[task_name]
        if task["total"] == 0:
            return 0.0
        return task["success"] / task["total"]

    def report(self):
        """Print metrics report."""
        print("\n── Metrics ────────────────────────────────────────────────────")
        for task_name, metrics in self.tasks.items():
            success_rate = self.get_success_rate(task_name)
            print(f"{task_name}:")
            print(f"  Total attempts: {metrics['total']}")
            print(f"  Success rate: {success_rate:.1%}")
            if metrics['validation_errors'] > 0:
                print(f"  Validation errors: {metrics['validation_errors']}")
            if metrics['json_errors'] > 0:
                print(f"  JSON errors: {metrics['json_errors']}")
            if metrics['other_errors'] > 0:
                print(f"  Other errors: {metrics['other_errors']}")

            if success_rate < 0.8:
                print(f"  ⚠️  Low success rate")


metrics = ExtractionMetrics()


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM CALLING: Support both Claude and Groq
# ─────────────────────────────────────────────────────────────────────────────

def call_claude(prompt: str, model: str = "claude-3-5-sonnet-20241022") -> str:
    """Call Claude API and return response text."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Install anthropic: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def call_groq(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """Call Groq API and return response text."""
    import requests

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
        },
    )

    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def call_llm(prompt: str, provider: str = "claude") -> str:
    """Call LLM (Claude preferred, Groq fallback)."""
    if provider == "claude":
        try:
            return call_claude(prompt)
        except (ImportError, ValueError) as e:
            print(f"Claude not available: {e}")
            print("Falling back to Groq...")
            return call_groq(prompt)
    else:
        return call_groq(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRUCTURED EXTRACTION WITH RETRY & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def create_schema_prompt(schema: BaseModel, task: str) -> str:
    """Create a prompt that requests JSON output matching the schema."""
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    return f"""{task}

Respond with ONLY valid JSON matching this schema (no markdown, no extra text):

{schema_json}"""


def extract_and_validate(
    prompt: str,
    schema: BaseModel,
    task_name: str = "extraction",
    max_retries: int = 3,
    provider: str = "claude"
) -> tuple[bool, BaseModel | str]:
    """Extract structured output with retry and validation."""

    metrics.record_attempt(task_name)

    for attempt in range(max_retries):
        try:
            # Call LLM
            response = call_llm(prompt, provider=provider)

            # Extract JSON (handle markdown wrapping)
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                response = json_match.group()

            # Parse JSON
            raw_output = json.loads(response)

            # Validate against Pydantic schema
            validated = schema(**raw_output)
            metrics.record_success(task_name)
            return True, validated

        except json.JSONDecodeError as e:
            metrics.record_json_error(task_name)
            if attempt == max_retries - 1:
                return False, f"Invalid JSON from LLM: {e}"

        except ValidationError as e:
            metrics.record_validation_error(task_name)
            if attempt == max_retries - 1:
                return False, f"Validation failed: {e}"

        except Exception as e:
            metrics.record_other_error(task_name)
            if attempt == max_retries - 1:
                return False, f"Error: {e}"

        # Retry with backoff
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"  Retry {attempt + 1}/{max_retries - 1} in {wait_time}s...")
            time.sleep(wait_time)

    return False, "Max retries exceeded"


# ─────────────────────────────────────────────────────────────────────────────
# DEMOS
# ─────────────────────────────────────────────────────────────────────────────

def demo_mock_extraction() -> None:
    """Demo: mock extraction showing structured output."""
    print("\n── Demo: Contact Extraction (Mock) ───────────────────────────")

    text = """Hi, my name is Alice Johnson and I work at TechCorp as an engineer.
    My email is alice@techcorp.com. Please also reach out to Bob Smith
    (bob.smith@company.org) from the product team."""

    print(f"\nInput text:\n  {text.strip()[:80]}...")

    # Mock LLM response
    mock_response = {
        "text": text,
        "extracted": [
            {
                "name": "Alice Johnson",
                "email": "alice@techcorp.com",
                "company": "TechCorp",
                "role": "Engineer",
            },
            {
                "name": "Bob Smith",
                "email": "bob.smith@company.org",
                "company": None,
                "role": "Product",
            },
        ],
        "count": 2,
    }

    print("\nValidating mock response against schema...")
    try:
        result = ExtractionResult(**mock_response)
        print(f"\n✓ Extraction succeeded ({result.count} contacts found)")
        for contact in result.extracted:
            print(f"  - {contact.name} ({contact.email})")
            if contact.company:
                print(f"    Company: {contact.company}, Role: {contact.role}")
    except ValidationError as e:
        print(f"\n✗ Validation failed: {e}")


def demo_contact_extraction() -> None:
    """Demo: extract contacts from text (requires API key)."""
    print("\n── Demo: Contact Extraction (Live API) ───────────────────────")

    text = """
    Hi, my name is Alice Johnson and I work at TechCorp as an engineer.
    My email is alice@techcorp.com. Please also reach out to Bob Smith
    (bob.smith@company.org) from the product team.
    """

    print(f"\nInput text:\n  {text.strip()[:100]}...")

    task = f"""Extract all contacts from this text.

Text: {text}"""

    prompt = create_schema_prompt(ExtractionResult, task)

    print("\nCalling LLM (with 3 retries)...")

    success, result = extract_and_validate(
        prompt,
        ExtractionResult,
        task_name="contact_extraction"
    )

    if success:
        print(f"\n✓ Extraction succeeded ({result.count} contacts found)")
        for contact in result.extracted:
            print(f"  - {contact.name} ({contact.email})")
            if contact.company:
                print(f"    Company: {contact.company}")
    else:
        print(f"\n✗ Extraction failed: {result}")


def demo_sentiment_analysis() -> None:
    """Demo: analyze sentiment with structured output."""
    print("\n── Demo: Sentiment Analysis ──────────────────────────────────")

    text = "I absolutely love this product! It makes my life so much easier."

    print(f"\nInput: '{text}'")

    task = f"""Analyze the sentiment of this text:

Text: {text}"""

    prompt = create_schema_prompt(SentimentAnalysis, task)

    print("Expected: sentiment (positive/negative/neutral) + confidence + explanation")
    print("Calling LLM (with 3 retries)...")

    success, result = extract_and_validate(
        prompt,
        SentimentAnalysis,
        task_name="sentiment_analysis"
    )

    if success:
        print(f"\n✓ Analysis succeeded")
        print(f"  Sentiment: {result.sentiment}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Explanation: {result.explanation}")
    else:
        print(f"\n✗ Analysis failed: {result}")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_tests() -> None:
    """Test structured output validation."""
    print("\n── Tests ────────────────────────────────────────────────────")
    failures = 0

    # Test 1: Valid Contact passes validation
    try:
        contact = Contact(
            name="Alice",
            email="alice@example.com",
            company="TechCorp",
            role="Engineer",
        )
        print("  ✓  Valid contact passes Pydantic validation")
    except Exception as e:
        print(f"  ✗  FAIL: Valid contact rejected: {e}")
        failures += 1

    # Test 2: Invalid email is rejected
    try:
        Contact(name="Bob", email="not-an-email", company=None, role=None)
        print("  ✗  FAIL: Invalid email was accepted")
        failures += 1
    except ValueError:
        print("  ✓  Invalid email is rejected")

    # Test 3: Valid SentimentAnalysis passes validation
    try:
        sentiment = SentimentAnalysis(
            text="Great product!",
            sentiment="positive",
            confidence=0.95,
            explanation="Uses positive words like 'great'",
        )
        print("  ✓  Valid sentiment analysis passes validation")
    except Exception as e:
        print(f"  ✗  FAIL: Valid sentiment rejected: {e}")
        failures += 1

    # Test 4: Invalid sentiment is rejected
    try:
        SentimentAnalysis(
            text="Test",
            sentiment="unknown",
            confidence=0.5,
            explanation="Test",
        )
        print("  ✗  FAIL: Invalid sentiment was accepted")
        failures += 1
    except ValueError:
        print("  ✓  Invalid sentiment is rejected")

    # Test 5: Invalid confidence is rejected
    try:
        SentimentAnalysis(
            text="Test",
            sentiment="positive",
            confidence=1.5,  # out of range
            explanation="Test",
        )
        print("  ✗  FAIL: Out-of-range confidence was accepted")
        failures += 1
    except ValueError:
        print("  ✓  Out-of-range confidence is rejected")

    # Test 6: Schema defines required fields
    try:
        Contact(name="Alice", email="alice@example.com")  # company and role are optional
        print("  ✓  Optional fields work correctly")
    except Exception as e:
        print(f"  ✗  FAIL: Optional fields rejected: {e}")
        failures += 1

    print()
    if failures == 0:
        print("  All tests passed ✓")
    else:
        print(f"  {failures} test(s) failed ✗")

    return failures == 0


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run tests (always works, no API needed)
    all_pass = run_tests()

    # Run mock demo (no API key needed)
    demo_mock_extraction()

    # Try live API demos
    print("\n" + "─" * 70)
    print("Attempting live API demos...\n")

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("GROQ_API_KEY")

    if api_key:
        try:
            demo_contact_extraction()
        except Exception as e:
            print(f"  Demo skipped: {e}")

        try:
            demo_sentiment_analysis()
        except Exception as e:
            print(f"  Demo skipped: {e}")

        # Report metrics
        metrics.report()
    else:
        print("No API keys found.")
        print("To enable live demos, set one of:")
        print("  export ANTHROPIC_API_KEY=your_key")
        print("  export GROQ_API_KEY=your_key")
