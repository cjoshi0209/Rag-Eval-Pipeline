"""Optional LLM-as-judge scorer. Only used when USE_OPENAI=true and
OPENAI_API_KEY is set -- otherwise the offline lexical scorer handles
everything (see evaluator.py).
"""
from __future__ import annotations

import json
from typing import List, Tuple


def score_with_openai(api_key: str, query: str, context: List[str], response: str) -> Tuple[float, float, float, list]:
    from openai import OpenAI  # imported lazily so it's an optional dep

    client = OpenAI(api_key=api_key)
    context_block = "\n".join(context)
    prompt = (
        "You are grading a RAG system's answer. Return three floats 0-1 "
        "(faithfulness, answer_relevance, context_recall) as JSON, plus a "
        "list of any response sentences not supported by the context.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer: {response}"
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(completion.choices[0].message.content)
    return (
        float(data.get("faithfulness", 0.0)),
        float(data.get("answer_relevance", 0.0)),
        float(data.get("context_recall", 0.0)),
        list(data.get("unsupported_claims", [])),
    )
