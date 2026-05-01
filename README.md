# LLM Engineering

Hands-on implementations of the core concepts every AI/ML engineer needs to know.
Each module is a self-contained Python script with explanations in the code.

## Modules

| # | Topic | Key concepts |
|---|---|---|
| [01 Embeddings](01_embeddings/) | Turning text into vectors | Semantic similarity, cosine distance, embedding models |
| [02 RAG](02_rag/) | Retrieval-Augmented Generation | Chunking, vector stores, retrieval, augmented generation |
| [03 Structured Outputs](03_structured_outputs/) | Getting reliable structure from LLMs | Pydantic, function calling, JSON mode |
| [04 Agents](04_agents/) | LLMs that take actions | Tool use, multi-step reasoning, loops |
| [05 Evaluation](05_evaluation/) | Measuring LLM quality | LLM-as-a-judge, metrics, label-free eval |
| [06 Fine-tuning](06_fine_tuning/) | Adapting models to specific tasks | When to fine-tune, LoRA, synthetic data |

## Setup

```bash
git clone https://github.com/franciscoambrosio/llm-engineering
cd llm-engineering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add your API keys to .env
```

Each module can be run independently:
```bash
python 01_embeddings/embeddings.py
```

## API keys

Modules use [Groq](https://console.groq.com) (free tier) by default.
Some modules optionally support the Anthropic API for comparison.
