from types import SimpleNamespace

import pytest

import agents
from market_data import MarketDataError, NewsDataError


def test_judge_rejects_malformed_model_output(monkeypatch):
    monkeypatch.setattr(agents, "chat", lambda *args, **kwargs: "not JSON")
    state = {
        "ticker": "AAPL",
        "constraint": "long term",
        "market_data": "price",
        "news": [],
        "bull_argument": "bull",
        "bear_attack": "bear",
        "bull_defense": "reply",
        "bear_defends": "reply",
    }
    with pytest.raises(agents.ModelOutputError, match="invalid decision"):
        agents.judge(state)


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
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
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
    assert agents.chat("prompt", structured=True) == "answer"
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


def test_context_provider_failure_is_not_silenced(monkeypatch):
    def fail(_ticker):
        raise MarketDataError("price unavailable")

    monkeypatch.setattr(agents, "get_market_data", fail)
    with pytest.raises(MarketDataError, match="price unavailable"):
        agents.user_input({"raw_ticker": "aapl", "constraint": "long term"})

    monkeypatch.setattr(agents, "get_market_data", lambda _ticker: "price")
    monkeypatch.setattr(
        agents,
        "get_headlines",
        lambda _ticker: (_ for _ in ()).throw(NewsDataError("news unavailable")),
    )
    with pytest.raises(NewsDataError, match="news unavailable"):
        agents.user_input({"raw_ticker": "aapl", "constraint": "long term"})
