# -*- coding: utf-8 -*-
"""
Module 05 — Evaluation
======================

Problem: You've built an LLM system. How do you know if it works?

Without evaluation:
  "Our chatbot seems good!"
  Problem: "Seems" is not data. You don't know:
    - How often it fails
    - Which types of queries it struggles with
    - If it's better/worse than alternatives
    - If changes improve or hurt performance

With evaluation:
  "Our chatbot has 92% accuracy on 1000 test cases"
  Benefit: Measurable, repeatable, comparable

This module covers:
  1. Test cases: what to test
  2. Metrics: how to measure (accuracy, latency, cost)
  3. Baselines: what's "good enough"
  4. Human evaluation: when numbers aren't enough
  5. Automated scoring: grade responses without humans

Why it matters:
  - Data-driven decisions: improve with evidence
  - Regression detection: catch when you break something
  - Stakeholder trust: "it works" → "it works 92% of the time"
  - ROI: understand cost vs benefit

Run:
  python 05_evaluation/evaluation.py
"""

import json
import time
import os
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict

# Load environment variables
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                if key not in os.environ:
                    os.environ[key] = value
except:
    pass

# LangFuse integration (optional)
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """A test case: input and expected output."""
    id: str
    input: str
    expected_output: str
    category: str  # e.g., "math", "factual", "reasoning"
    difficulty: str  # e.g., "easy", "medium", "hard"


# Example test cases for a weather chatbot
TEST_CASES = [
    TestCase(
        id="math_1",
        input="What is 2 + 2?",
        expected_output="4",
        category="math",
        difficulty="easy"
    ),
    TestCase(
        id="math_2",
        input="Calculate 15 * 8",
        expected_output="120",
        category="math",
        difficulty="easy"
    ),
    TestCase(
        id="fact_1",
        input="What is the capital of France?",
        expected_output="Paris",
        category="factual",
        difficulty="easy"
    ),
    TestCase(
        id="fact_2",
        input="Who wrote Romeo and Juliet?",
        expected_output="Shakespeare",
        category="factual",
        difficulty="easy"
    ),
    TestCase(
        id="reason_1",
        input="If all dogs are animals, and Fido is a dog, is Fido an animal?",
        expected_output="Yes",
        category="reasoning",
        difficulty="medium"
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MOCK SYSTEM (replace with real LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

def system_response(query: str) -> str:
    """Mock LLM system. Returns a response to the query."""
    # In reality, this would call your LLM
    # For demo, we mock some responses

    if "2 + 2" in query:
        return "4"  # Correct
    elif "15 * 8" in query:
        return "120"  # Correct
    elif "capital of France" in query:
        return "Paris"  # Correct
    elif "Romeo and Juliet" in query:
        return "William Shakespeare"  # Correct (longer form)
    elif "Fido" in query:
        return "No"  # WRONG! Should be "Yes"
    else:
        return "I don't know"


# ─────────────────────────────────────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Results from evaluating one test case."""
    test_id: str
    category: str
    input: str
    expected: str
    actual: str
    correct: bool
    latency_ms: float
    score: float  # 0.0-1.0


def exact_match(expected: str, actual: str) -> bool:
    """Exact match scoring."""
    return expected.lower().strip() == actual.lower().strip()


def contains_match(expected: str, actual: str) -> bool:
    """Partial match: does response contain the expected answer?"""
    return expected.lower() in actual.lower()


def similarity_match(expected: str, actual: str, threshold: float = 0.8) -> float:
    """Simple word overlap scoring (0.0-1.0)."""
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())

    if not expected_words:
        return 1.0 if not actual_words else 0.0

    overlap = len(expected_words & actual_words)
    union = len(expected_words | actual_words)

    score = overlap / union if union > 0 else 0.0
    return 1.0 if score >= threshold else score


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Run tests and measure performance. Optionally integrate with LangFuse."""

    def __init__(self, scoring_fn=exact_match, use_langfuse: bool = False):
        self.scoring_fn = scoring_fn
        self.results: List[EvaluationResult] = []
        self.langfuse = None
        self.trace = None

        # Initialize LangFuse if available and requested
        if use_langfuse and LANGFUSE_AVAILABLE:
            try:
                self.langfuse = Langfuse()
                print("✓ LangFuse connected")
            except Exception as e:
                print(f"⚠️  LangFuse setup failed: {e}")
                print("   Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY env vars")
        elif use_langfuse and not LANGFUSE_AVAILABLE:
            print("⚠️  LangFuse not installed. Install with: pip install langfuse")

    def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate one test case. Log to LangFuse if connected."""
        start = time.time()
        actual = system_response(test_case.input)
        latency_ms = (time.time() - start) * 1000

        # Score: 1.0 if correct, 0.0 if wrong
        correct = self.scoring_fn(test_case.expected_output, actual)
        score = 1.0 if correct else 0.0

        result = EvaluationResult(
            test_id=test_case.id,
            category=test_case.category,
            input=test_case.input,
            expected=test_case.expected_output,
            actual=actual,
            correct=correct,
            latency_ms=latency_ms,
            score=score
        )

        # Log to LangFuse if connected
        if self.langfuse:
            # Create an event for the test
            event = self.langfuse.create_event(
                name=f"test_{test_case.id}",
                input={"question": test_case.input},
                output={"answer": actual, "expected": test_case.expected_output},
                metadata={
                    "category": test_case.category,
                    "difficulty": test_case.difficulty,
                    "latency_ms": latency_ms
                }
            )

            # Score the event (this shows up in analytics!)
            self.langfuse.create_score(
                trace_id=event.trace_id if hasattr(event, 'trace_id') else "",
                name="accuracy",
                value=score,
                comment=f"{'✓ Correct' if correct else '✗ Wrong'}: {test_case.category}"
            )

        self.results.append(result)
        return result

    def run_all(self, test_cases: List[TestCase], run_name: str = "evaluation") -> None:
        """Evaluate all test cases. Optionally trace with LangFuse."""
        for test_case in test_cases:
            self.evaluate(test_case)

        if self.langfuse:
            # Log summary to LangFuse
            summary = self.summary()
            self.langfuse.create_event(
                name=f"{run_name}_summary",
                input={"test_count": summary["total_tests"]},
                output={
                    "accuracy": summary["accuracy"],
                    "total_tests": summary["total_tests"],
                    "passed": summary["passed"]
                }
            )
            # Flush to ensure data is sent
            self.langfuse.flush()
            print(f"\n📊 Evaluation logged to LangFuse!")
            print(f"View at: https://cloud.langfuse.com/dashboard")

    def summary(self) -> Dict:
        """Compute summary metrics."""
        if not self.results:
            return {}

        total = len(self.results)
        passed = sum(1 for r in self.results if r.correct)
        accuracy = passed / total if total > 0 else 0.0
        avg_latency = sum(r.latency_ms for r in self.results) / total

        # By category
        by_category = {}
        for category in set(r.category for r in self.results):
            cat_results = [r for r in self.results if r.category == category]
            cat_passed = sum(1 for r in cat_results if r.correct)
            cat_accuracy = cat_passed / len(cat_results) if cat_results else 0.0
            by_category[category] = {
                "total": len(cat_results),
                "passed": cat_passed,
                "accuracy": cat_accuracy
            }

        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "by_category": by_category
        }

    def report(self) -> str:
        """Print a human-readable report."""
        summary = self.summary()

        report = "\n── Evaluation Report ────────────────────────────────\n"
        report += f"Total tests: {summary['total_tests']}\n"
        report += f"Passed: {summary['passed']}\n"
        report += f"Failed: {summary['failed']}\n"
        report += f"Accuracy: {summary['accuracy']:.1%}\n"
        report += f"Avg latency: {summary['avg_latency_ms']:.1f}ms\n"

        report += "\n── By Category ──────────────────────────────────\n"
        for category, stats in summary["by_category"].items():
            report += f"{category}: {stats['accuracy']:.1%} ({stats['passed']}/{stats['total']})\n"

        report += "\n── Failures ──────────────────────────────────────\n"
        failures = [r for r in self.results if not r.correct]
        if failures:
            for f in failures:
                report += f"\n  {f.test_id} ({f.category}):\n"
                report += f"    Input: {f.input}\n"
                report += f"    Expected: {f.expected}\n"
                report += f"    Got: {f.actual}\n"
        else:
            report += "  (none)\n"

        return report


# ─────────────────────────────────────────────────────────────────────────────
# 5. BASELINES & COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

class Baseline:
    """A baseline result to compare against."""

    def __init__(self, name: str, accuracy: float, latency_ms: float):
        self.name = name
        self.accuracy = accuracy
        self.latency_ms = latency_ms

    def compare(self, accuracy: float, latency_ms: float) -> str:
        """Compare current results to baseline."""
        acc_change = (accuracy - self.accuracy) * 100
        lat_change = latency_ms - self.latency_ms

        acc_emoji = "📈" if acc_change > 0 else "📉" if acc_change < 0 else "➡️"
        lat_emoji = "⚡" if lat_change < 0 else "🐢" if lat_change > 0 else "➡️"

        result = f"\nComparison to '{self.name}' baseline:\n"
        result += f"  Accuracy: {acc_emoji} {acc_change:+.1f}% (was {self.accuracy:.1%})\n"
        result += f"  Latency: {lat_emoji} {lat_change:+.1f}ms (was {self.latency_ms:.1f}ms)\n"

        return result


# ─────────────────────────────────────────────────────────────────────────────
# TESTS & DEMOS
# ─────────────────────────────────────────────────────────────────────────────

def demo_evaluation():
    """Run evaluation demo."""
    print("\n── Running Evaluation ───────────────────────────────")
    print(f"Testing {len(TEST_CASES)} cases...\n")

    evaluator = Evaluator(scoring_fn=exact_match)
    evaluator.run_all(TEST_CASES)

    print(evaluator.report())

    # Compare to baseline
    baseline = Baseline(name="v1.0", accuracy=0.90, latency_ms=150)
    summary = evaluator.summary()
    print(baseline.compare(summary["accuracy"], summary["avg_latency_ms"]))


def demo_baselines():
    """Show how to track performance over time."""
    print("\n── Tracking Performance Over Time ──────────────────")

    baselines = [
        Baseline(name="v1.0", accuracy=0.80, latency_ms=200),
        Baseline(name="v1.1", accuracy=0.85, latency_ms=180),
        Baseline(name="v2.0", accuracy=0.92, latency_ms=150),
    ]

    print("\nVersion history:")
    for baseline in baselines:
        print(f"  {baseline.name}: {baseline.accuracy:.1%} accuracy, {baseline.latency_ms:.0f}ms")

    print(f"\nCurrent system improving towards v2.0...")


def demo_scoring_methods():
    """Show different scoring approaches."""
    print("\n── Scoring Methods ──────────────────────────────────")

    test_case = TestCase(
        id="test_1",
        input="Who wrote Romeo and Juliet?",
        expected_output="Shakespeare",
        category="factual",
        difficulty="easy"
    )

    actual = "William Shakespeare wrote Romeo and Juliet"

    print(f"\nTest: {test_case.input}")
    print(f"Expected: {test_case.expected_output}")
    print(f"Actual: {actual}\n")

    exact = exact_match(test_case.expected_output, actual)
    contains = contains_match(test_case.expected_output, actual)
    similarity = similarity_match(test_case.expected_output, actual)

    print(f"Exact match: {exact} (too strict)")
    print(f"Contains match: {contains} (better)")
    print(f"Similarity score: {similarity:.2f} (flexible)")


def test_langfuse_connection():
    """Test if LangFuse is properly configured."""
    print("\n── Testing LangFuse Connection ──────────────────────")

    if not LANGFUSE_AVAILABLE:
        print("❌ LangFuse not installed")
        print("   Install with: pip install langfuse")
        return False

    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")

    if not secret_key:
        print("❌ LANGFUSE_SECRET_KEY not set")
        print("   Get from: https://cloud.langfuse.com → Settings → API Keys")
        return False

    if not public_key:
        print("❌ LANGFUSE_PUBLIC_KEY not set")
        print("   Get from: https://cloud.langfuse.com → Settings → API Keys")
        return False

    print("✓ LangFuse installed")
    print(f"✓ LANGFUSE_SECRET_KEY set (starts with {secret_key[:10]}...)")
    print(f"✓ LANGFUSE_PUBLIC_KEY set (starts with {public_key[:10]}...)")

    try:
        client = Langfuse()
        print("✓ Connected to LangFuse!")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def demo_with_langfuse():
    """Run evaluation with LangFuse tracing."""
    print("\n── Evaluation with LangFuse ────────────────────────")

    # First test connection
    if not test_langfuse_connection():
        return

    print("\nRunning evaluation with tracing...")
    evaluator = Evaluator(scoring_fn=exact_match, use_langfuse=True)
    evaluator.run_all(TEST_CASES, run_name="module05_demo")

    print("\n✅ Evaluation complete!")
    print("📊 View traces at: https://cloud.langfuse.com → Projects → Traces")


if __name__ == "__main__":
    print("=" * 60)
    print("Module 05: Evaluation")
    print("=" * 60)

    demo_evaluation()
    demo_scoring_methods()
    demo_baselines()

    # Uncomment to trace with LangFuse
    # demo_with_langfuse()
