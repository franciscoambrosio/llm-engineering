"""
Module 02 — Retrieval-Augmented Generation (RAG)
================================================

RAG is a pattern for giving LLMs access to external knowledge without fine-tuning:
  1. Chunk: break documents into pieces
  2. Embed: turn chunks into vectors
  3. Store: index them for fast retrieval
  4. Retrieve: find chunks relevant to a query (semantic search)
  5. Augment: add them to the LLM prompt as context
  6. Generate: let the LLM answer with grounded knowledge

Why RAG?
- LLMs have a knowledge cutoff (training data ends at a date)
- LLMs hallucinate on unfamiliar or niche topics
- RAG grounds answers in real documents → fewer hallucinations

This module uses:
  - sentence-transformers for embeddings (same as Module 01)
  - a simple in-memory vector store (no external DB)
  - Groq for LLM generation (free tier)

Run:
  python 02_rag/rag.py
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

MODEL = "all-MiniLM-L6-v2"

# Knowledge base: facts about AI/ML researchers (source: public knowledge)
DOCUMENTS = [
    "Yann LeCun is the Chief AI Scientist at Meta. He pioneered convolutional neural networks (CNNs) and won the Turing Award in 2018.",
    "Geoffrey Hinton is known as one of the pioneers of deep learning. He invented backpropagation and founded the Vector Institute at University of Toronto.",
    "Yoshua Bengio is a deep learning pioneer and Turing Award winner. He founded MILA and focuses on AI safety and causal representation learning.",
    "Andrew Ng founded Coursera and deeplearning.ai. He teaches machine learning and advocates for AI education and responsible AI development.",
    "Fei-Fei Li is the Co-Director of the Stanford HAI institute. She focuses on human-centered AI and founded SAIL (Stanford AI Lab).",
    "Demis Hassabis is the CEO and co-founder of DeepMind. DeepMind created AlphaGo, AlphaFold, and other breakthrough AI systems.",
    "OpenAI was founded in 2015 by Sam Altman, Elon Musk, and others. It created GPT-4, ChatGPT, and DALL-E.",
    "Anthropic was founded in 2021 by former OpenAI researchers including Dario and Daniela Amodei. It focuses on AI safety and created Claude.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Range: [-1, 1]."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
# In practice: split on sentences, paragraphs, or use sliding windows.
# For this module: each document is already a single chunk (simplification).

def chunk_documents(docs: list[str], chunk_size: int = 1) -> list[str]:
    """Split documents into chunks. Here: 1 chunk per document (no splitting)."""
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# 2. VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────
# In practice: use Pinecone, Weaviate, or Postgres+pgvector.
# Here: simple in-memory dict.

class SimpleVectorStore:
    """Minimal vector store: stores chunks + embeddings in memory."""

    def __init__(self):
        self.chunks = []  # original text
        self.embeddings = []  # np arrays

    def add(self, chunks: list[str], embeddings: list[np.ndarray]):
        """Add chunks and their embeddings."""
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Find top-k chunks by similarity to query embedding."""
        scores = [
            (cosine_similarity(query_embedding, emb), chunk)
            for chunk, emb in zip(self.chunks, self.embeddings)
        ]
        scores.sort(reverse=True)
        return [(chunk, score) for score, chunk in scores[:k]]


# ─────────────────────────────────────────────────────────────────────────────
# 3. RAG PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_pipeline(docs: list[str], model: SentenceTransformer) -> SimpleVectorStore:
    """Build the RAG pipeline: chunk → embed → store."""
    chunks = chunk_documents(docs)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    store = SimpleVectorStore()
    store.add(chunks, embeddings)
    return store


def retrieve(query: str, store: SimpleVectorStore, model: SentenceTransformer, k: int = 3) -> list[str]:
    """Retrieve top-k relevant chunks for a query."""
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    results = store.search(query_embedding, k=k)
    return [chunk for chunk, _ in results]


def augment_prompt(query: str, context: list[str]) -> str:
    """Add retrieved context to the query."""
    context_str = "\n".join([f"- {chunk}" for chunk in context])
    return f"""You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context_str}

Question: {query}

Answer:"""


def generate_with_rag(query: str, store: SimpleVectorStore, model: SentenceTransformer) -> str:
    """Full RAG pipeline: retrieve → augment → generate."""
    context = retrieve(query, store, model, k=3)
    prompt = augment_prompt(query, context)

    # Generate with Groq (free, fast)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fallback: just show the augmented prompt
        return prompt

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
            },
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Generation failed ({e}). Showing augmented prompt instead:\n")
        return prompt

    return "Generation failed. Check GROQ_API_KEY."


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

def demo_basic_rag(model: SentenceTransformer) -> None:
    print("\n── Basic RAG Pipeline ────────────────────────────────────────")

    store = build_rag_pipeline(DOCUMENTS, model)

    query = "Who founded DeepMind?"
    print(f"\nQuery: '{query}'")
    print("\nRetrieved context:")
    context = retrieve(query, store, model, k=2)
    for i, chunk in enumerate(context, 1):
        print(f"  {i}. {chunk[:80]}...")

    print("\nAugmented prompt sent to LLM:")
    prompt = augment_prompt(query, context)
    print(f"  {prompt[:200]}...")


def demo_rag_quality(model: SentenceTransformer) -> None:
    print("\n── RAG Quality: Retrieval Relevance ──────────────────────────")

    store = build_rag_pipeline(DOCUMENTS, model)

    test_queries = [
        ("Who created ChatGPT?", 1),  # expected: OpenAI doc
        ("What is MILA?", 1),  # expected: Bengio doc
        ("Who focuses on AI safety?", 2),  # multiple docs match
    ]

    for query, expected_matches in test_queries:
        query_emb = model.encode([query], convert_to_numpy=True)[0]
        results = store.search(query_emb, k=3)

        print(f"\n  Query: '{query}'")
        relevant = 0
        for chunk, score in results:
            preview = chunk[:70]
            if score > 0.4:  # threshold: relevant
                print(f"    ✓ {score:.3f}  {preview}...")
                relevant += 1
            else:
                print(f"      {score:.3f}  {preview}...")

        status = "✓" if relevant >= expected_matches else "✗"
        print(f"  {status}  Found {relevant}/{expected_matches} relevant chunks")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(model: SentenceTransformer) -> None:
    print("\n── Tests ────────────────────────────────────────────────────")
    failures = 0

    # Test 1: Vector store stores chunks correctly
    store = SimpleVectorStore()
    chunks = ["Chunk A", "Chunk B"]
    embs = [np.array([0.1, 0.2]), np.array([0.15, 0.25])]
    store.add(chunks, embs)

    if len(store.chunks) == 2 and len(store.embeddings) == 2:
        print("  ✓  Vector store stores chunks and embeddings")
    else:
        print("  ✗  FAIL: Vector store didn't store correctly")
        failures += 1

    # Test 2: Retrieval returns relevant chunks
    store = build_rag_pipeline(DOCUMENTS, model)
    query = "Who founded Anthropic?"
    context = retrieve(query, store, model, k=1)

    if context and "Anthropic" in context[0]:
        print("  ✓  Retrieval returns relevant chunks (keyword match)")
    else:
        print("  ✗  FAIL: Retrieval missed relevant chunk")
        failures += 1

    # Test 3: Augmented prompt includes context
    context = ["Fact 1", "Fact 2"]
    prompt = augment_prompt("Test query?", context)

    if "Fact 1" in prompt and "Fact 2" in prompt and "Test query?" in prompt:
        print("  ✓  Augmented prompt includes context and query")
    else:
        print("  ✗  FAIL: Augmented prompt malformed")
        failures += 1

    # Test 4: Similar queries retrieve overlapping results
    store = build_rag_pipeline(DOCUMENTS, model)
    results_1 = set(retrieve("Who invented CNNs?", store, model, k=2))
    results_2 = set(retrieve("LeCun and convolutional networks", store, model, k=2))
    overlap = results_1 & results_2

    if len(overlap) > 0:
        print("  ✓  Similar queries return overlapping results")
    else:
        print("  ✗  FAIL: Similar queries returned different results")
        failures += 1

    print()
    if failures == 0:
        print("  All tests passed.")
    else:
        print(f"  {failures} test(s) failed.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading model...")
    model = SentenceTransformer(MODEL)

    demo_basic_rag(model)
    demo_rag_quality(model)
    run_tests(model)
