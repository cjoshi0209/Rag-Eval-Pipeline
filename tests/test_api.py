import os
import tempfile

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mktemp(suffix='.db')}"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_evaluate_and_metrics_roundtrip():
    with TestClient(app) as client:
        payload = {
            "query": "What is the refund policy?",
            "context": ["Refunds within 30 days with a receipt."],
            "response": "You can get a refund within 30 days if you have a receipt.",
        }
        resp = client.post("/evaluate", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "faithfulness" in body
        assert "hallucination_detected" in body

        metrics = client.get("/metrics").json()
        assert metrics["total_evaluations"] >= 1

        hist = client.get("/metrics/history").json()
        assert len(hist) >= 1
