from types import SimpleNamespace

import pytest

import graph
from schemas import Evidence


def offline_graph(monkeypatch, confidences):
    calls = {"judge": 0, "critic": 0}
    monkeypatch.setattr(graph, "get_settings", lambda: SimpleNamespace(retry_threshold=0.6))
    monkeypatch.setattr(
        graph,
        "collect_context",
        lambda state: {
            "ticker": "AAPL",
            "evidence": [Evidence(id="price_1", source="price", text="price summary")],
            "retry_count": 0,
        },
    )
    monkeypatch.setattr(graph, "bull_analysis", lambda state: {"bull_argument": "bull"})
    monkeypatch.setattr(graph, "bear_critique", lambda state: {"bear_critique": "bear"})
    monkeypatch.setattr(graph, "bull_revision", lambda state: {"bull_revision": "revision"})

    def judge(state):
        calls["judge"] += 1
        return {
            "decision": "HOLD",
            "confidence": confidences[calls["judge"] - 1],
            "summary": "uncertain",
            "supporting_evidence": ["price_1"],
            "risks": [],
            "limitations": [],
        }

    def critic(state):
        calls["critic"] += 1
        return {
            "critic_feedback": "Reconsider uncertainty",
            "retry_count": state["retry_count"] + 1,
        }

    monkeypatch.setattr(graph, "judge", judge)
    monkeypatch.setattr(graph, "critic", critic)
    return graph.build_graph(), calls


@pytest.mark.parametrize("confidence", [0.6, 0.9])
def test_sufficient_confidence_skips_critic(monkeypatch, confidence):
    debate, calls = offline_graph(monkeypatch, [confidence])
    result = debate.invoke({"raw_ticker": "AAPL", "constraint": "long term", "retry_count": 0})
    assert result["retry_count"] == 0
    assert calls == {"judge": 1, "critic": 0}


@pytest.mark.parametrize("final_confidence", [0.2, 0.8])
def test_low_confidence_uses_exactly_one_retry(monkeypatch, final_confidence):
    debate, calls = offline_graph(monkeypatch, [0.3, final_confidence])
    result = debate.invoke({"raw_ticker": "AAPL", "constraint": "long term", "retry_count": 0})
    assert result["retry_count"] == 1
    assert result["confidence"] == final_confidence
    assert calls == {"judge": 2, "critic": 1}
