"""
Core evaluation logic for the RAG Evaluation Pipeline.

Runs fully offline by default using lightweight lexical/statistical scoring
(no paid API required), so it works out of the box. Set USE_OPENAI=true and
provide OPENAI_API_KEY to swap in an LLM-as-judge scorer for production use.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from rag_eval.lexical import cosine_similarity, estimate_tokens, term_vector


@dataclass
class EvaluationResult:
    query: str
    faithfulness: float
    answer_relevance: float
    context_recall: float
    hallucination_detected: bool
    latency_ms: float
    token_cost_usd: float
    unsupported_claims: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_recall": round(self.context_recall, 4),
            "hallucination_detected": self.hallucination_detected,
            "latency_ms": round(self.latency_ms, 2),
            "token_cost_usd": round(self.token_cost_usd, 6),
            "unsupported_claims": self.unsupported_claims,
        }


class Evaluator:
    """Scores an (query, retrieved context, LLM response) triple.

    Thresholds mirror the ones documented in the README:
      faithfulness > 0.85, answer_relevance > 0.80, context_recall > 0.75,
      hallucination_rate < 0.10.
    """

    FAITHFULNESS_THRESHOLD = 0.55  # lexical-overlap scale; tune per corpus
    PRICE_PER_1K_PROMPT_TOKENS = 0.005   # GPT-4o-mini-ish placeholder pricing
    PRICE_PER_1K_COMPLETION_TOKENS = 0.015

    def __init__(self, api_key: Optional[str] = None, use_openai: Optional[bool] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.use_openai = (
            use_openai if use_openai is not None
            else os.getenv("USE_OPENAI", "false").lower() == "true"
        )

    def evaluate(self, query: str, context: List[str], response: str) -> EvaluationResult:
        start = time.perf_counter()

        if self.use_openai and self.api_key:
            from rag_eval.openai_judge import score_with_openai

            faithfulness, relevance, recall, unsupported = score_with_openai(
                self.api_key, query, context, response
            )
        else:
            faithfulness, relevance, recall, unsupported = self._score_offline(
                query, context, response
            )

        latency_ms = (time.perf_counter() - start) * 1000
        cost = self._estimate_cost(query, context, response)

        return EvaluationResult(
            query=query,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            context_recall=recall,
            hallucination_detected=faithfulness < self.FAITHFULNESS_THRESHOLD,
            latency_ms=latency_ms,
            token_cost_usd=cost,
            unsupported_claims=unsupported,
        )

    # -- offline scoring (default, no external calls) -----------------

    def _score_offline(self, query, context, response):
        response_vec = term_vector(response)
        query_vec = term_vector(query)
        context_text = " ".join(context)
        context_vec = term_vector(context_text)

        faithfulness = cosine_similarity(response_vec, context_vec)
        relevance = cosine_similarity(response_vec, query_vec)
        recall = cosine_similarity(context_vec, query_vec)

        # Flag response sentences that share almost no vocabulary with any
        # retrieved chunk -- a cheap proxy for "unsupported claim".
        unsupported = []
        for sentence in re.split(r"(?<=[.!?])\s+", response.strip()):
            if not sentence:
                continue
            sent_vec = term_vector(sentence)
            best = max((cosine_similarity(sent_vec, term_vector(c)) for c in context), default=0.0)
            if best < 0.15 and len(sent_vec) >= 3:
                unsupported.append(sentence.strip())

        return faithfulness, relevance, recall, unsupported

    def _estimate_cost(self, query, context, response) -> float:
        prompt_tokens = estimate_tokens(query + " ".join(context))
        completion_tokens = estimate_tokens(response)
        return (
            prompt_tokens / 1000 * self.PRICE_PER_1K_PROMPT_TOKENS
            + completion_tokens / 1000 * self.PRICE_PER_1K_COMPLETION_TOKENS
        )
