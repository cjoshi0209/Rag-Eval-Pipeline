<div align="center">

# RAG Evaluation Pipeline

**Production-grade LLM evaluation system with hallucination scoring, context faithfulness tracking, and real-time observability.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-7C3AED?style=flat-square)](LICENSE)

[Overview](#-overview) · [Architecture](#-architecture) · [Metrics](#-evaluation-metrics) · [Quick Start](#-quick-start) · [API Docs](#-api-reference)

</div>

---

## Overview

Most RAG systems ship without any quality gate. Answers look plausible but silently hallucinate, drift off context, or degrade as the knowledge base grows. This pipeline gives you **observable, measurable RAG quality** — the same evaluation infrastructure used at ORIXEN.AI for client deployments.

**What this solves:**
- Catching hallucinations before they reach users
- Tracking retrieval quality degradation over time
- Getting cost + latency visibility per query
- Providing a baseline to A/B test chunking strategies, embeddings, and prompts

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Eval Pipeline                     │
│                                                          │
│  Query ──► Retriever ──► Context ──► LLM ──► Response   │
│               │                              │           │
│               ▼                              ▼           │
│         [Retrieval Eval]            [Generation Eval]    │
│         · Context recall           · Faithfulness score  │
│         · MRR / Hit@K              · Answer relevance    │
│         · Latency                  · Hallucination flag  │
│               │                              │           │
│               └──────────► Metrics DB ◄──────┘           │
│                                  │                       │
│                            Dashboard UI                  │
└─────────────────────────────────────────────────────────┘
```

---

## Evaluation Metrics

| Metric | Description | Threshold |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? | > 0.85 |
| **Answer Relevance** | Does the answer address the question? | > 0.80 |
| **Context Recall** | Did retrieval surface the right chunks? | > 0.75 |
| **Hallucination Rate** | Claims not supported by context | < 0.10 |
| **Latency P95** | End-to-end response time | < 2000ms |
| **Token Cost** | Cost per query (GPT-4 pricing) | Tracked |

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/cjoshi0209/rag-eval-pipeline
cd rag-eval-pipeline

# Environment
cp .env.example .env
# Add OPENAI_API_KEY, DATABASE_URL

# Docker (recommended)
docker-compose up -d

# Or local
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API is live at `http://localhost:8000` · Docs at `http://localhost:8000/docs`

---

## Usage

```python
from rag_eval import Evaluator

evaluator = Evaluator(api_key="sk-...")

result = evaluator.evaluate(
    query="What is our refund policy?",
    context=retrieved_chunks,
    response=llm_response
)

print(result)
# {
#   "faithfulness": 0.92,
#   "answer_relevance": 0.88,
#   "hallucination_detected": False,
#   "latency_ms": 847,
#   "token_cost_usd": 0.0023
# }
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/evaluate` | POST | Run full evaluation on a query-response pair |
| `/metrics` | GET | Retrieve aggregated metrics dashboard data |
| `/metrics/history` | GET | Time-series quality trend data |
| `/health` | GET | Service health check |

---

## Stack

`Python 3.11` · `FastAPI` · `LangChain` · `OpenAI GPT-4` · `PostgreSQL` · `Redis` · `Docker` · `Prometheus`

---

## Related

Built as part of the evaluation infrastructure at [ORIXEN.AI](https://orixenai.in). If you're building production RAG systems and want to talk evaluation strategy, reach out on [LinkedIn](https://linkedin.com/in/chinmay-joshi-62b873212).

---

<div align="center">
Built by <a href="https://joshichinmay.tech">Chinmay Joshi</a> · <a href="https://orixenai.in">ORIXEN.AI</a>
</div>
