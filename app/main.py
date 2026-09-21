from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.store import aggregate_metrics, history, init_db, record_evaluation
from rag_eval.evaluator import Evaluator

evaluator = Evaluator()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="RAG Evaluation Pipeline",
    description="Hallucination scoring, faithfulness, and retrieval-quality "
    "observability for RAG systems.",
    version="0.1.0",
    lifespan=lifespan,
)


class EvaluateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context: List[str] = Field(..., min_length=1)
    response: str = Field(..., min_length=1)


@app.post("/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    result = evaluator.evaluate(req.query, req.context, req.response)
    payload = result.to_dict()
    record_evaluation(payload)
    return payload


@app.get("/metrics")
def metrics() -> dict:
    return aggregate_metrics()


@app.get("/metrics/history")
def metrics_history(limit: int = 100) -> list:
    return history(limit=limit)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
