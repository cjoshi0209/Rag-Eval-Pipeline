"""Lightweight lexical scoring helpers.

Pure-Python term-frequency vectors + cosine similarity. No external
dependencies, so the pipeline runs fully offline by default.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "and", "or", "but", "if", "it",
    "this", "that", "these", "those", "as", "at", "by", "from", "we", "you",
    "i", "our", "your", "their", "he", "she", "do", "does", "did", "not",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def term_vector(text: str) -> Counter:
    return Counter(tokenize(text))


def cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def estimate_tokens(text: str) -> int:
    # Rough heuristic (~4 chars/token) used when tiktoken isn't installed.
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)
