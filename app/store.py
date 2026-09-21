"""Tiny SQLite-backed metrics store (stands in for the Metrics DB in the
architecture diagram). Swap for Postgres in production by pointing
DATABASE_URL elsewhere and replacing the two functions below.
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Iterator

DB_PATH = os.getenv("DATABASE_URL", "sqlite:///./eval.db").replace("sqlite:///", "")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    faithfulness REAL NOT NULL,
    answer_relevance REAL NOT NULL,
    context_recall REAL NOT NULL,
    hallucination_detected INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    token_cost_usd REAL NOT NULL
);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)


def record_evaluation(result: dict) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO evaluations
               (ts, query, faithfulness, answer_relevance, context_recall,
                hallucination_detected, latency_ms, token_cost_usd)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                result["query"],
                result["faithfulness"],
                result["answer_relevance"],
                result["context_recall"],
                int(result["hallucination_detected"]),
                result["latency_ms"],
                result["token_cost_usd"],
            ),
        )


def aggregate_metrics() -> dict:
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n,
                      AVG(faithfulness) AS avg_faithfulness,
                      AVG(answer_relevance) AS avg_relevance,
                      AVG(context_recall) AS avg_recall,
                      AVG(latency_ms) AS avg_latency_ms,
                      SUM(token_cost_usd) AS total_cost_usd,
                      SUM(hallucination_detected) AS hallucination_count
               FROM evaluations"""
        ).fetchone()
    n = row["n"] or 0
    return {
        "total_evaluations": n,
        "avg_faithfulness": round(row["avg_faithfulness"], 4) if n else None,
        "avg_answer_relevance": round(row["avg_relevance"], 4) if n else None,
        "avg_context_recall": round(row["avg_recall"], 4) if n else None,
        "avg_latency_ms": round(row["avg_latency_ms"], 2) if n else None,
        "total_cost_usd": round(row["total_cost_usd"], 4) if n else 0,
        "hallucination_rate": round(row["hallucination_count"] / n, 4) if n else None,
    }


def history(limit: int = 100) -> list:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT ts, query, faithfulness, answer_relevance, context_recall,
                      hallucination_detected, latency_ms, token_cost_usd
               FROM evaluations ORDER BY ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
