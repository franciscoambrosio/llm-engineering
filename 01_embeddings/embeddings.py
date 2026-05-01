"""
Module 01 — Embeddings
======================
An embedding is a vector (list of numbers) that represents the *meaning* of a piece of text.
Similar meanings → vectors that are close together in space.

This module covers:
  1. Computing embeddings with a local model
  2. Measuring semantic similarity with cosine distance
  3. Semantic search — finding relevant documents without keyword matching
  4. Clustering — grouping texts by topic without labels

Run:
  python 01_embeddings/embeddings.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# We use a local model — no API key needed.
# "all-MiniLM-L6-v2": small (80MB), fast, strong baseline for most tasks.
MODEL = "all-MiniLM-L6-v2"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Range: [-1, 1], higher = more similar."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. WHAT IS AN EMBEDDING?
# ─────────────────────────────────────────────────────────────────────────────

def demo_embeddings(model: SentenceTransformer) -> None:
    print("\n── 1. What is an embedding? ─────────────────────────────────")

    sentences = [
        "The cat sat on the mat.",
        "A kitten rested on the rug.",      # same meaning, different words
        "The stock market crashed today.",  # unrelated topic
    ]

    embeddings = model.encode(sentences)
    print(f"  Model output shape: {embeddings.shape}")
    print(f"  Each sentence → vector of {embeddings.shape[1]} numbers")
    print()

    sim_same  = cosine_similarity(embeddings[0], embeddings[1])
    sim_diff  = cosine_similarity(embeddings[0], embeddings[2])

    print(f"  '{sentences[0]}'")
    print(f"    vs '{sentences[1]}'  →  similarity: {sim_same:.3f}  (same meaning)")
    print(f"    vs '{sentences[2]}'  →  similarity: {sim_diff:.3f}  (different topic)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEMANTIC SEARCH
# ─────────────────────────────────────────────────────────────────────────────
# Keyword search finds the word "python" → misses "serpent" or "snake".
# Semantic search finds documents about the *concept*, regardless of exact words.

def demo_semantic_search(model: SentenceTransformer) -> None:
    print("\n── 2. Semantic search ───────────────────────────────────────")

    documents = [
        "How to reverse a linked list in Python",
        "Understanding gradient descent in neural networks",
        "Best practices for REST API design",
        "Introduction to backpropagation",
        "Database indexing strategies for performance",
        "Overfitting and regularization in machine learning",
        "How HTTP requests work under the hood",
        "Loss functions: MSE, cross-entropy, and when to use each",
    ]

    query = "how do neural networks learn?"

    doc_embeddings   = model.encode(documents)
    query_embedding  = model.encode([query])[0]

    scores = [(cosine_similarity(query_embedding, e), doc) for e, doc in zip(doc_embeddings, documents)]
    ranked = sorted(scores, reverse=True)

    print(f"  Query: '{query}'")
    print(f"\n  Results (ranked by semantic similarity):")
    for score, doc in ranked:
        marker = "✓" if score > 0.35 else " "
        print(f"    {marker} {score:.3f}  {doc}")

    print()
    print("  Note: 'backpropagation' and 'gradient descent' rank high even though")
    print("  the query doesn't contain those words. Keyword search would miss these.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLUSTERING BY TOPIC
# ─────────────────────────────────────────────────────────────────────────────
# Embeddings can group texts by meaning with no labels — purely from geometry.

def demo_clustering(model: SentenceTransformer) -> None:
    print("\n── 3. Clustering by topic ───────────────────────────────────")

    texts = [
        # Group A — machine learning
        "Stochastic gradient descent converges faster with momentum.",
        "Dropout is a regularization technique to prevent overfitting.",
        "Transformers use self-attention to model token relationships.",
        # Group B — cooking
        "Sauté the onions until golden brown before adding garlic.",
        "Pasta should be cooked al dente, not soft.",
        "Reduce the sauce over medium heat for 10 minutes.",
        # Group C — personal finance
        "Diversify your portfolio to reduce exposure to single assets.",
        "Compound interest grows wealth exponentially over time.",
        "Index funds outperform most actively managed funds long-term.",
    ]

    embeddings = model.encode(texts)

    # Compute all pairwise similarities
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            pairs.append((sim, i, j, texts[i], texts[j]))

    pairs.sort(reverse=True)

    print("  Most similar pairs (should always be within the same group):")
    for sim, i, j, a, b in pairs[:6]:
        same_group = (i // 3) == (j // 3)
        marker = "✓" if same_group else "✗"
        print(f"    {marker} {sim:.3f}  '{a[:45]}...'")
        print(f"           '{b[:45]}...'")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_tests(model: SentenceTransformer) -> None:
    print("\n── Tests ────────────────────────────────────────────────────")
    failures = 0

    # Test 1: similar sentences score higher than unrelated ones
    cat     = model.encode(["The cat sat on the mat."])[0]
    kitten  = model.encode(["A kitten rested on the rug."])[0]
    stocks  = model.encode(["The stock market crashed today."])[0]

    sim_close = cosine_similarity(cat, kitten)
    sim_far   = cosine_similarity(cat, stocks)

    if sim_close > sim_far:
        print(f"  ✓  Similar sentences score higher than unrelated ones ({sim_close:.3f} > {sim_far:.3f})")
    else:
        print(f"  ✗  FAIL: expected {sim_close:.3f} > {sim_far:.3f}")
        failures += 1

    # Test 2: a sentence is identical to itself (similarity = 1.0)
    vec = model.encode(["Hello world."])[0]
    self_sim = cosine_similarity(vec, vec)
    if abs(self_sim - 1.0) < 1e-5:
        print(f"  ✓  Self-similarity is 1.0 ({self_sim:.6f})")
    else:
        print(f"  ✗  FAIL: self-similarity should be 1.0, got {self_sim:.6f}")
        failures += 1

    # Test 3: semantic search returns relevant result in top-2
    documents = [
        "How to cook pasta",
        "Gradient descent optimizes neural networks",
        "The French Revolution began in 1789",
        "Backpropagation computes gradients layer by layer",
    ]
    query = "how do neural networks learn?"
    doc_embs   = model.encode(documents)
    query_emb  = model.encode([query])[0]
    scores     = [cosine_similarity(query_emb, e) for e in doc_embs]
    top2_idxs  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
    # indices 1 and 3 are the ML-related documents
    if 1 in top2_idxs and 3 in top2_idxs:
        print(f"  ✓  Semantic search returns both ML documents in top-2")
    else:
        print(f"  ✗  FAIL: expected indices 1 and 3 in top-2, got {top2_idxs}")
        failures += 1

    # Test 4: within-group similarity > cross-group similarity (clustering)
    ml_a    = model.encode(["Dropout prevents overfitting in deep learning."])[0]
    ml_b    = model.encode(["Batch normalization stabilizes training."])[0]
    cooking = model.encode(["Boil the pasta for 8 minutes."])[0]

    within = cosine_similarity(ml_a, ml_b)
    cross  = cosine_similarity(ml_a, cooking)
    if within > cross:
        print(f"  ✓  Within-group similarity > cross-group ({within:.3f} > {cross:.3f})")
    else:
        print(f"  ✗  FAIL: expected within-group ({within:.3f}) > cross-group ({cross:.3f})")
        failures += 1

    print()
    if failures == 0:
        print(f"  All tests passed.")
    else:
        print(f"  {failures} test(s) failed.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading model...")
    model = SentenceTransformer(MODEL)

    demo_embeddings(model)
    demo_semantic_search(model)
    demo_clustering(model)
    run_tests(model)
