# RAG Evaluation Pipeline

Lightweight LLM evaluation system for RAG pipelines — hallucination scoring, context faithfulness tracking, and retrieval-quality observability. Runs fully offline by default (no paid API key required); an optional GPT-4o-mini judge can be enabled for higher-fidelity scoring.

Overview · Architecture · Metrics · Quick Start · API Docs

## Overview

Most RAG systems ship without any quality gate. Answers look plausible but silently hallucinate, drift off context, or degrade as the knowledge base grows. This pipeline gives you an observable, measurable RAG quality signal you can wire into CI or a dashboard.

What this solves:
- Catching hallucinations before they reach users
- Tracking retrieval quality degradation over time
- Getting cost + latency visibility per query
- Providing a baseline to A/B test chunking strategies, embeddings, and prompts

## Architecture

```
  Query --> Retriever --> Context --> LLM --> Response
               |                        |
               v                        v
        [Retrieval Eval]         [Generation Eval]
        - Context recall         - Faithfulness score
        - Cosine similarity      - Answer relevance
        - Latency                - Hallucination flag
               |                        |
               +---------> SQLite Metrics DB <---------+
                                  |
                          /metrics endpoint
```

Two scoring backends, selected by `USE_OPENAI`:
- **Offline (default)** — pure-Python TF-IDF-style term vectors + cosine similarity (`rag_eval/lexical.py`). No dependencies, no API key, deterministic.
- **LLM-as-judge (optional)** — GPT-4o-mini grades the same triple when `USE_OPENAI=true` and `OPENAI_API_KEY` is set (`rag_eval/openai_judge.py`).

## Evaluation Metrics

| Metric | Description |
|---|---|
| Faithfulness | Is the response grounded in the retrieved context? (cosine similarity between response and context term vectors) |
| Answer Relevance | Does the response address the query? |
| Context Recall | Did retrieval surface chunks relevant to the query? |
| Hallucination Detected | `faithfulness` below a configurable threshold |
| Latency (ms) | Wall-clock time for the scoring call itself |
| Token Cost (USD) | Estimated from token count × configurable per-1K pricing |

The offline scorer's thresholds are tuned for lexical overlap, not semantic similarity — swap in the OpenAI judge (or a real embedding model) for production-grade accuracy; the interface (`Evaluator.evaluate`) stays the same either way.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/cjoshi0209/Rag-Eval-Pipeline
cd Rag-Eval-Pipeline

# Environment
cp .env.example .env
# Optional: set OPENAI_API_KEY + USE_OPENAI=true for the LLM-as-judge scorer

# Docker (recommended)
docker-compose up -d

# Or local
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API is live at http://localhost:8000 · Docs at http://localhost:8000/docs

Run the test suite:

```bash
pip install -r requirements.txt
pytest -v
```

## Usage

```python
from rag_eval import Evaluator

evaluator = Evaluator()  # offline by default; pass use_openai=True + api_key for GPT-4o-mini judge

result = evaluator.evaluate(
    query="What is our refund policy?",
    context=["Refunds are processed within 30 days of purchase with a receipt."],
    response="You can get a refund within 30 days if you have a receipt.",
)

print(result.to_dict())
# {
#   "query": "What is our refund policy?",
#   "faithfulness": 0.83,
#   "answer_relevance": 0.71,
#   "context_recall": 0.65,
#   "hallucination_detected": False,
#   "latency_ms": 0.04,
#   "token_cost_usd": 0.000123,
#   "unsupported_claims": []
# }
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/evaluate` | POST | Run evaluation on a query/context/response triple, persist it |
| `/metrics` | GET | Aggregated metrics across all recorded evaluations |
| `/metrics/history` | GET | Most recent evaluations, newest first |
| `/health` | GET | Service health check |

## Stack

Python 3.11 · FastAPI · SQLite · Docker · pytest · (optional) OpenAI GPT-4o-mini

## Related

Built as part of the evaluation infrastructure work at ORIXEN.AI. If you're building production RAG systems and want to talk evaluation strategy, reach out on LinkedIn.

Built by Chinmay Joshi · ORIXEN.AI
