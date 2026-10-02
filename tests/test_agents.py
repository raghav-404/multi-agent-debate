from types import SimpleNamespace

import pytest

import agents
from market_data import MarketDataError, NewsDataError
from schemas import Evidence


def judge_state():
    return {
        "ticker": "AAPL",
        "constraint": "long term",
        "evidence": [Evidence(id="price_1", source="price", text="last close 100")],
        "bull_argument": "bull",
        "bear_critique": "bear",
        "bull_revision": "reply",
        "retry_count": 0,
    }


def test_judge_rejects_malformed_model_output(monkeypatch):
    monkeypatch.setattr(agents, "chat", lambda *args, **kwargs: agents.ModelReply("not JSON"))
    with pytest.raises(agents.ModelOutputError, match="invalid decision"):
        agents.judge(judge_state())


def test_judge_requires_real_evidence_ids(monkeypatch):
    monkeypatch.setattr(
        agents,
        "chat",
        lambda *args, **kwargs: agents.ModelReply(
            '{"decision":"HOLD","confidence":0.8,"summary":"Reason",'
            '"supporting_evidence":["news_9"],"risks":[],"limitations":[]}'
        ),
    )
    with pytest.raises(agents.ModelOutputError, match="unknown evidence IDs: news_9"):
        agents.judge(judge_state())


def test_judge_accepts_retrieved_evidence_id(monkeypatch):
    monkeypatch.setattr(
        agents,
        "chat",
        lambda *args, **kwargs: agents.ModelReply(
            '{"decision":"HOLD","confidence":0.8,"summary":"Reason",'
            '"supporting_evidence":["price_1"],"risks":[],"limitations":[]}'
        ),
    )
    assert agents.judge(judge_state())["supporting_evidence"] == ["price_1"]


def test_final_judge_receives_critic_feedback(monkeypatch):
    prompts = []

    def fake_chat(prompt, **kwargs):
        prompts.append(prompt)
        return agents.ModelReply(
            '{"decision":"HOLD","confidence":0.7,"summary":"Revised",'
            '"supporting_evidence":["price_1"],"risks":[],"limitations":[]}'
        )

    monkeypatch.setattr(agents, "chat", fake_chat)
    agents.judge(judge_state() | {"critic_feedback": "Missing downside evidence", "retry_count": 1})
    assert "Missing downside evidence" in prompts[0]


def test_missing_key_is_an_explicit_failure(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(agents.ModelError, match="GROQ_API_KEY"):
        agents.chat("prompt")


def test_chat_uses_configured_groq_model_without_network(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
            )

    class FakeGroq:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(agents, "Groq", FakeGroq)
    monkeypatch.setattr(
        agents,
        "get_settings",
        lambda: SimpleNamespace(groq_api_key="test-key", groq_model="test-model"),
    )
    reply = agents.chat("prompt", structured=True)
    assert reply.content == "answer"
    assert (reply.prompt_tokens, reply.completion_tokens) == (12, 4)
    assert calls[0]["model"] == "test-model"
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_chat_provider_failure_is_explicit(monkeypatch):
    class FailedGroq:
        def __init__(self, **kwargs):
            raise RuntimeError("offline")

    monkeypatch.setattr(agents, "Groq", FailedGroq)
    monkeypatch.setattr(
        agents,
        "get_settings",
        lambda: SimpleNamespace(groq_api_key="test-key", groq_model="test-model"),
    )
    with pytest.raises(agents.ModelError, match="Groq request failed"):
        agents.chat("prompt")


def test_empty_structured_response_is_a_judge_failure(monkeypatch):
    class EmptyGroq:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **options: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
                    )
                )
            )

    monkeypatch.setattr(agents, "Groq", EmptyGroq)
    monkeypatch.setattr(
        agents,
        "get_settings",
        lambda: SimpleNamespace(groq_api_key="test-key", groq_model="test-model"),
    )
    with pytest.raises(agents.JudgeOutputError, match="empty response"):
        agents.chat("prompt", structured=True)


def test_context_provider_failure_is_not_silenced(monkeypatch):
    def fail(_ticker):
        raise MarketDataError("price unavailable")

    monkeypatch.setattr(agents, "get_market_data", fail)
    with pytest.raises(MarketDataError, match="price unavailable"):
        agents.collect_context({"raw_ticker": "aapl", "constraint": "long term"})

    monkeypatch.setattr(agents, "get_market_data", lambda _ticker: "price")
    monkeypatch.setattr(
        agents,
        "get_headlines",
        lambda _ticker: (_ for _ in ()).throw(NewsDataError("news unavailable")),
    )
    with pytest.raises(NewsDataError, match="news unavailable"):
        agents.collect_context({"raw_ticker": "aapl", "constraint": "long term"})


def test_context_assigns_stable_request_evidence_ids(monkeypatch):
    monkeypatch.setattr(agents, "get_market_data", lambda ticker: "price summary")
    monkeypatch.setattr(agents, "get_headlines", lambda ticker: ["first", "second"])
    result = agents.collect_context({"raw_ticker": " aapl ", "constraint": "long term"})
    assert result["ticker"] == "AAPL"
    assert [item.id for item in result["evidence"]] == ["price_1", "news_1", "news_2"]
    assert result["retry_count"] == 0


def test_context_uses_provided_evidence_without_provider_calls(monkeypatch):
    def provider_must_not_run(_ticker):
        raise AssertionError("provider called despite supplied context")

    monkeypatch.setattr(agents, "get_market_data", provider_must_not_run)
    monkeypatch.setattr(agents, "get_headlines", provider_must_not_run)
    evidence = [Evidence(id="price_1", source="price", text="snapshot")]
    result = agents.collect_context(
        {"raw_ticker": "AAPL", "constraint": "long term", "provided_evidence": evidence}
    )
    assert result["evidence"] == evidence


def test_critic_refuses_second_retry():
    state = judge_state() | {
        "retry_count": 1,
        "decision": "HOLD",
        "confidence": 0.3,
        "summary": "x",
    }
    with pytest.raises(agents.ModelError, match="only once"):
        agents.critic(state)
