# Module 05: Evaluation

How to measure if your LLM system actually works.

## Run the Evaluation

```bash
python evaluation.py
```

## LangFuse Integration

This module supports **LangFuse** for production-grade observability.

### What is LangFuse?

LangFuse is an observability platform for LLM applications. It automatically tracks:
- **Traces**: Full execution flow
- **Latency**: How long requests take
- **Costs**: How much you spend per request
- **Evals**: Custom evaluation scores
- **User feedback**: Thumbs up/down from users

### Setup LangFuse (Optional)

1. **Create free account** at https://cloud.langfuse.com

2. **Install SDK**:
   ```bash
   pip install langfuse
   ```

3. **Get API keys** from LangFuse dashboard (Settings → API Keys)

4. **Set environment variables**:
   ```bash
   export LANGFUSE_SECRET_KEY=sk_prod_...
   export LANGFUSE_PUBLIC_KEY=pk_prod_...
   ```

5. **Enable in code** by uncommenting in `evaluation.py`:
   ```python
   demo_with_langfuse()
   ```

### What Gets Traced?

When you run with LangFuse enabled:

```
📊 Trace: "evaluation"
  └─ Span: "test_math_1"
     ├─ Input: "What is 2 + 2?"
     ├─ Output: "4"
     ├─ Score: 1.0 ✓
     └─ Metadata: {"category": "math", "latency_ms": 0.5}
  
  └─ Span: "test_fact_2"
     ├─ Input: "Who wrote Romeo and Juliet?"
     ├─ Output: "William Shakespeare"
     ├─ Score: 0.0 ✗
     └─ Metadata: {"category": "factual", "latency_ms": 0.2}
```

Then view at: https://cloud.langfuse.com → Project → Traces

### Development vs Production

| Phase | Use LangFuse? | Why |
|-------|---------------|-----|
| Local development | No | Too much setup, use simple eval |
| Before deployment | Optional | Want baseline metrics |
| Production | Yes | Track real users, costs, feedback |

### Example: Production Workflow

```python
# Before deploy: Run module 05 evaluation
evaluator = Evaluator()
evaluator.run_all(TEST_CASES)
assert evaluator.summary()["accuracy"] > 0.90

# Deploy system to production

# In production: Trace with LangFuse
evaluator = Evaluator(use_langfuse=True)
result = evaluator.evaluate(user_query)
# Automatically logged to LangFuse dashboard
```

## Scoring Methods

### Exact Match (Strictest)
```python
Expected: "Shakespeare"
Actual:   "William Shakespeare"
Result:   ❌ FAIL
```
Good for: Math, facts with one answer

### Contains (Moderate)
```python
Expected: "Shakespeare"
Actual:   "William Shakespeare wrote Romeo"
Result:   ✅ PASS (contains "shakespeare")
```
Good for: When answer is buried in longer text

### Similarity (Lenient)
```python
Expected: "Shakespeare"
Actual:   "William Shakespeare"
Result:   ~0.5 (partial match)
```
Good for: Fuzzy matching, partial credit

## Key Concepts

### Test Cases
- **Input**: What you ask the system
- **Expected**: The correct answer (ground truth)
- **Category**: Type of question (math, facts, reasoning)
- **Difficulty**: How hard (easy, medium, hard)

### Evaluation Metrics
- **Accuracy**: % of correct answers
- **Latency**: How fast it responds
- **By Category**: Accuracy broken down by type
- **Failures**: Which tests failed and why

### Baselines
- Know if you're improving
- Track progress over time
- Compare versions (v1.0 → v2.0)

## Example Output

```
── Evaluation Report ────────────────────────────────
Total tests: 5
Passed: 3
Failed: 2
Accuracy: 60.0%

── By Category ──────────────────────────────────
math: 100% (2/2)
factual: 50% (1/2)
reasoning: 0% (0/1)

── Failures ──────────────────────────────────────
fact_2: Expected "Shakespeare", Got "William Shakespeare"
reason_1: Expected "Yes", Got "No"
```

## Next Steps

- Run evaluation.py locally
- (Optional) Set up LangFuse for dashboard view
- Move to Module 06: Fine-tuning
