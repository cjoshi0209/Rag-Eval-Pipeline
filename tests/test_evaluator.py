from rag_eval.evaluator import Evaluator


def test_grounded_response_scores_high_faithfulness():
    ev = Evaluator(use_openai=False)
    context = ["Our refund policy allows returns within 30 days of purchase with a receipt."]
    result = ev.evaluate(
        query="What is the refund policy?",
        context=context,
        response="Refunds are allowed within 30 days of purchase if you have a receipt.",
    )
    assert result.faithfulness > 0.4
    assert result.hallucination_detected is False


def test_ungrounded_response_flagged_as_hallucination():
    ev = Evaluator(use_openai=False)
    context = ["Our refund policy allows returns within 30 days of purchase with a receipt."]
    result = ev.evaluate(
        query="What is the refund policy?",
        context=context,
        response="We offer free lifetime warranty and same-day international shipping on all orders.",
    )
    assert result.faithfulness < 0.4
    assert result.hallucination_detected is True


def test_cost_and_latency_are_populated():
    ev = Evaluator(use_openai=False)
    result = ev.evaluate("q", ["some context"], "some response")
    assert result.latency_ms >= 0
    assert result.token_cost_usd >= 0
