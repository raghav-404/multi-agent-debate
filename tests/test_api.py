from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api
import service
from agents import ModelError
from market_data import MarketDataError, NewsDataError
from schemas import DebateResponse, Evidence
from service import ConfigurationError, PersistenceError

client = TestClient(api.app)


def valid_response():
    return DebateResponse(
        ticker="AAPL",
        constraint="long term",
        decision="HOLD",
        confidence=0.45,
        summary="mixed signals",
        supporting_evidence=["price_1"],
        evidence_used=[Evidence(id="price_1", source="price", text="price summary")],
        risks=["volatility"],
        limitations=[],
        latency_ms=120,
        retry_count=1,
        model_name="test-model",
        persisted=False,
    )


def test_health_is_lightweight(monkeypatch):
    monkeypatch.setattr(api, "run_debate", lambda request: (_ for _ in ()).throw(AssertionError()))
    assert client.get("/health").json() == {"status": "ok"}


def test_post_debates_success(monkeypatch):
    def run(request):
        assert request.ticker == "AAPL"
        return valid_response()

    monkeypatch.setattr(api, "run_debate", run)
    response = client.post("/debates", json={"ticker": " aapl ", "constraint": "long term"})
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "HOLD"
    assert body["retry_count"] == 1
    assert body["evidence_used"][0]["id"] == "price_1"
    assert body["persisted"] is False


def test_post_debates_runs_shared_service_offline(monkeypatch):
    class FakeGraph:
        def invoke(self, state):
            assert state["raw_ticker"] == "AAPL"
            return {
                "ticker": "AAPL",
                "constraint": "long term",
                "decision": "HOLD",
                "confidence": 0.7,
                "summary": "mixed signals",
                "supporting_evidence": ["price_1"],
                "evidence": [Evidence(id="price_1", source="price", text="price summary")],
                "risks": [],
                "limitations": [],
                "retry_count": 0,
            }

    monkeypatch.setattr(service, "build_graph", FakeGraph)
    monkeypatch.setattr(
        service,
        "get_settings",
        lambda: SimpleNamespace(groq_api_key="test", groq_model="model", database_url=None),
    )
    response = client.post("/debates", json={"ticker": "aapl", "constraint": "long term"})
    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"
    assert response.json()["persisted"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"ticker": "bad ticker", "constraint": "long term"},
        {"ticker": "AAPL", "constraint": " "},
        {"ticker": "AAPL"},
    ],
)
def test_post_debates_validation_error(payload):
    assert client.post("/debates", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ConfigurationError("GROQ_API_KEY is required"), 503),
        (MarketDataError("Price provider failed"), 502),
        (NewsDataError("News provider failed"), 502),
        (ModelError("Groq request failed"), 502),
        (PersistenceError("Decision could not be stored"), 503),
        (RuntimeError("unexpected"), 500),
    ],
)
def test_post_debates_failure_responses(monkeypatch, error, status):
    def fail(request):
        raise error

    monkeypatch.setattr(api, "run_debate", fail)
    response = client.post("/debates", json={"ticker": "AAPL", "constraint": "long term"})
    assert response.status_code == status
    assert "decision" not in response.json()
    if status == 500:
        assert response.json()["detail"] == "Unexpected internal failure"
