from types import SimpleNamespace

import main
from schemas import DebateRequest


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
