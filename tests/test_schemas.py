import pytest
from pydantic import ValidationError

from schemas import DebateRequest, Evidence, JudgeDecision


def test_ticker_is_normalized_without_guessing_a_different_symbol():
    assert DebateRequest(ticker=" aapl ", constraint=" long term ").ticker == "AAPL"
    assert DebateRequest(ticker=" btc-usd ", constraint="long term").ticker == "BTC-USD"


@pytest.mark.parametrize("ticker", ["", "AAPL US", "AAPL/US", "A" * 16])
def test_invalid_ticker_is_rejected(ticker):
    with pytest.raises(ValidationError):
        DebateRequest(ticker=ticker, constraint="long term")


def test_empty_constraint_is_rejected():
    with pytest.raises(ValidationError):
        DebateRequest(ticker="AAPL", constraint="  ")


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_outside_unit_interval_is_rejected(confidence):
    with pytest.raises(ValidationError):
        JudgeDecision(
            decision="BUY", confidence=confidence, summary="Reason", supporting_evidence=["price_1"]
        )


def test_judge_decision_rejects_unknown_decision_and_extra_fields():
    with pytest.raises(ValidationError):
        JudgeDecision.model_validate_json(
            '{"decision":"NEUTRAL","confidence":0.5,"summary":"Reason",'
            '"supporting_evidence":["price_1"]}'
        )
    with pytest.raises(ValidationError):
        JudgeDecision.model_validate_json(
            '{"decision":"HOLD","confidence":0.5,"summary":"Reason",'
            '"supporting_evidence":["price_1"],"invented":1}'
        )


def test_judge_requires_evidence_citation():
    with pytest.raises(ValidationError):
        JudgeDecision(decision="HOLD", confidence=0.5, summary="Reason", supporting_evidence=[])


def test_evidence_has_request_level_id():
    assert Evidence(id="news_1", source="news", text="Headline").id == "news_1"
    with pytest.raises(ValidationError):
        Evidence(id="unknown", source="news", text="Headline")
