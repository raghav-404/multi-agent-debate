import main
from schemas import DebateRequest, DebateResponse, Evidence


def test_cli_uses_shared_service(monkeypatch, capsys):
    request = DebateRequest(ticker="aapl", constraint="long term")
    response = DebateResponse(
        ticker="AAPL",
        constraint="long term",
        decision="HOLD",
        confidence=0.4,
        summary="uncertain",
        supporting_evidence=["price_1"],
        evidence_used=[Evidence(id="price_1", source="price", text="price summary")],
        risks=["volatility"],
        limitations=[],
        latency_ms=100,
        retry_count=1,
        model_calls=6,
        model_name="model",
        persisted=False,
    )
    monkeypatch.setattr(main, "read_input", lambda: request)
    monkeypatch.setattr(main, "run_debate", lambda value: response if value is request else None)
    assert main.main() == 0
    output = capsys.readouterr().out
    assert "[price_1] price summary" in output
    assert "Critique retries: 1" in output
    assert "Not saved" in output
