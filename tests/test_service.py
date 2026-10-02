from types import SimpleNamespace

import psycopg
import pytest

import service
from schemas import DebateRequest, Evidence


def fake_graph_result():
    return {
        "ticker": "AAPL",
        "constraint": "long term",
        "decision": "HOLD",
        "confidence": 0.45,
        "summary": "mixed signals",
        "supporting_evidence": ["price_1"],
        "evidence": [
            Evidence(id="price_1", source="price", text="price summary"),
            Evidence(id="news_1", source="news", text="headline"),
        ],
        "risks": ["volatility"],
        "limitations": [],
        "retry_count": 1,
        "model_calls": 6,
        "token_reports": [],
    }


class FakeGraph:
    def invoke(self, state):
        assert state == {
            "raw_ticker": "AAPL",
            "constraint": "long term",
            "retry_count": 0,
            "model_calls": 0,
            "token_reports": [],
        }
        return fake_graph_result()


def settings(database_url=None, key="test-key"):
    return SimpleNamespace(
        groq_api_key=key,
        groq_model="test-model",
        database_url=database_url,
        retry_threshold=0.6,
    )


def test_service_returns_cited_evidence_without_database(monkeypatch):
    monkeypatch.setattr(service, "get_settings", lambda: settings())
    result = service.run_debate(
        DebateRequest(ticker="aapl", constraint="long term"), graph_runner=FakeGraph()
    )
    assert result.ticker == "AAPL"
    assert result.retry_count == 1
    assert result.model_calls == 6
    assert result.prompt_tokens is None
    assert result.persisted is False
    assert [item.id for item in result.evidence_used] == ["price_1"]


def test_service_reports_tokens_only_when_every_call_has_usage(monkeypatch):
    class UsageGraph:
        def invoke(self, state):
            return fake_graph_result() | {
                "model_calls": 2,
                "token_reports": [
                    {"prompt_tokens": 4, "completion_tokens": 2},
                    {"prompt_tokens": 6, "completion_tokens": 3},
                ],
            }

    monkeypatch.setattr(service, "get_settings", lambda: settings())
    result = service.run_debate(
        DebateRequest(ticker="AAPL", constraint="long term"), graph_runner=UsageGraph()
    )
    assert result.model_calls == 2
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


def test_evaluation_context_skips_database_write(monkeypatch):
    class SuppliedContextGraph:
        def invoke(self, state):
            assert [item.id for item in state["provided_evidence"]] == ["price_1"]
            return fake_graph_result()

    def storage_must_not_run(**kwargs):
        raise AssertionError("evaluation wrote a historical decision")

    monkeypatch.setattr(service, "get_settings", lambda: settings("postgresql://test"))
    monkeypatch.setattr(service, "save_run", storage_must_not_run)
    result = service.run_debate(
        DebateRequest(ticker="AAPL", constraint="long term"),
        graph_runner=SuppliedContextGraph(),
        evidence=[Evidence(id="price_1", source="price", text="snapshot")],
        persist=False,
    )
    assert result.persisted is False


def test_service_stores_final_decision(monkeypatch):
    saved = {}
    monkeypatch.setattr(service, "get_settings", lambda: settings("postgresql://test"))

    def save(**kwargs):
        saved.update(kwargs)
        return True

    monkeypatch.setattr(service, "save_run", save)
    result = service.run_debate(
        DebateRequest(ticker="aapl", constraint="long term"), graph_runner=FakeGraph()
    )
    assert result.persisted is True
    assert saved["ticker"] == "AAPL"
    assert saved["retry_count"] == 1
    assert saved["model_name"] == "test-model"


def test_service_fails_if_configured_database_is_unavailable(monkeypatch):
    monkeypatch.setattr(service, "get_settings", lambda: settings("postgresql://test"))

    def fail(**kwargs):
        raise psycopg.OperationalError("db offline")

    monkeypatch.setattr(service, "save_run", fail)
    with pytest.raises(service.PersistenceError, match="could not be stored"):
        service.run_debate(
            DebateRequest(ticker="AAPL", constraint="long term"), graph_runner=FakeGraph()
        )


def test_service_fails_if_configured_storage_returns_false(monkeypatch):
    monkeypatch.setattr(service, "get_settings", lambda: settings("postgresql://test"))
    monkeypatch.setattr(service, "save_run", lambda **kwargs: False)
    with pytest.raises(service.PersistenceError, match="could not be stored"):
        service.run_debate(
            DebateRequest(ticker="AAPL", constraint="long term"), graph_runner=FakeGraph()
        )


def test_service_rejects_missing_key_before_graph(monkeypatch):
    monkeypatch.setattr(service, "get_settings", lambda: settings(key=""))
    with pytest.raises(service.ConfigurationError, match="GROQ_API_KEY"):
        service.run_debate(
            DebateRequest(ticker="AAPL", constraint="long term"), graph_runner=FakeGraph()
        )


def test_service_rejects_invalid_configuration(monkeypatch):
    def bad_settings():
        raise ValueError("GROQ_MODEL must not be empty")

    monkeypatch.setattr(service, "get_settings", bad_settings)
    with pytest.raises(service.ConfigurationError, match="GROQ_MODEL"):
        service.run_debate(
            DebateRequest(ticker="AAPL", constraint="long term"), graph_runner=FakeGraph()
        )


def test_service_rejects_citation_missing_from_context(monkeypatch):
    class BadGraph:
        def invoke(self, state):
            return fake_graph_result() | {"supporting_evidence": ["news_9"]}

    monkeypatch.setattr(service, "get_settings", lambda: settings())
    with pytest.raises(service.EvidenceCitationError, match="evidence not present"):
        service.run_debate(
            DebateRequest(ticker="AAPL", constraint="long term"), graph_runner=BadGraph()
        )


def test_invalid_graph_decision_is_not_stored(monkeypatch):
    class BadGraph:
        def invoke(self, state):
            return fake_graph_result() | {"confidence": 1.5}

    def storage_must_not_run(**kwargs):
        raise AssertionError("invalid decision reached PostgreSQL")

    monkeypatch.setattr(service, "get_settings", lambda: settings("postgresql://test"))
    monkeypatch.setattr(service, "save_run", storage_must_not_run)
    with pytest.raises(service.JudgeOutputError, match="invalid decision"):
        service.run_debate(
            DebateRequest(ticker="AAPL", constraint="long term"), graph_runner=BadGraph()
        )
