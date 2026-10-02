from types import SimpleNamespace

import main
from schemas import DebateRequest, Evidence


def test_cli_missing_key_fails_before_graph_runs(monkeypatch, capsys):
    monkeypatch.setattr(
        main, "read_input", lambda: DebateRequest(ticker="AAPL", constraint="long term")
    )
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(groq_api_key=""))

    def graph_must_not_run():
        raise AssertionError("graph ran without an API key")

    monkeypatch.setattr(main, "build_graph", graph_must_not_run)
    assert main.main() == 1
    assert "GROQ_API_KEY is required" in capsys.readouterr().err


def test_cli_passes_retry_count_to_history_store(monkeypatch, capsys):
    saved = {}

    class FakeGraph:
        def invoke(self, state):
            assert state["retry_count"] == 0
            return {
                "ticker": "AAPL",
                "constraint": "long term",
                "decision": "HOLD",
                "confidence": 0.4,
                "summary": "uncertain",
                "supporting_evidence": ["price_1"],
                "evidence": [Evidence(id="price_1", source="price", text="price summary")],
                "risks": ["volatility"],
                "limitations": [],
                "retry_count": 1,
            }

    monkeypatch.setattr(
        main, "read_input", lambda: DebateRequest(ticker="AAPL", constraint="long term")
    )
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: SimpleNamespace(
            groq_api_key="test", database_url="postgresql://test", groq_model="model"
        ),
    )
    monkeypatch.setattr(main, "build_graph", FakeGraph)
    monkeypatch.setattr(main, "init_db", lambda: True)

    def save(**kwargs):
        saved.update(kwargs)
        return True

    monkeypatch.setattr(main, "save_run", save)
    assert main.main() == 0
    assert saved["retry_count"] == 1
    assert "[price_1] price summary" in capsys.readouterr().out
