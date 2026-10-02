import json
from types import SimpleNamespace

import pytest

import agents
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


@pytest.mark.parametrize("first_confidence, expected_calls", [(0.8, 4), (0.3, 6)])
def test_real_nodes_count_calls_and_usage_with_supplied_context(
    monkeypatch, first_confidence, expected_calls
):
    monkeypatch.setattr(graph, "get_settings", lambda: SimpleNamespace(retry_threshold=0.6))
    judge_calls = 0

    def fake_chat(prompt, *, structured=False):
        nonlocal judge_calls
        if structured:
            judge_calls += 1
        confidence = first_confidence if judge_calls == 1 else 0.8
        content = (
            json.dumps(
                {
                    "decision": "HOLD",
                    "confidence": confidence,
                    "summary": "mixed",
                    "supporting_evidence": ["price_1"],
                    "risks": [],
                    "limitations": [],
                }
            )
            if structured
            else "short reasoning [price_1]"
        )
        return agents.ModelReply(content, prompt_tokens=2, completion_tokens=3)

    monkeypatch.setattr(agents, "chat", fake_chat)
    evidence = [Evidence(id="price_1", source="price", text="snapshot")]
    result = graph.build_graph().invoke(
        {
            "raw_ticker": "AAPL",
            "constraint": "long term",
            "retry_count": 0,
            "model_calls": 0,
            "token_reports": [],
            "provided_evidence": evidence,
        }
    )
    assert result["model_calls"] == expected_calls
    assert len(result["token_reports"]) == expected_calls
    assert sum(report["prompt_tokens"] for report in result["token_reports"]) == 2 * expected_calls
    assert result["retry_count"] == (1 if expected_calls == 6 else 0)
