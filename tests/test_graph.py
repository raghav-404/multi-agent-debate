import graph


def test_linear_graph_runs_with_offline_nodes(monkeypatch):
    monkeypatch.setattr(
        graph,
        "user_input",
        lambda state: {"ticker": state["raw_ticker"], "market_data": "price", "news": []},
    )
    monkeypatch.setattr(graph, "bull", lambda state: {"bull_argument": "bull"})
    monkeypatch.setattr(graph, "bear_attack", lambda state: {"bear_attack": "bear"})
    monkeypatch.setattr(graph, "bull_defense", lambda state: {"bull_defense": "reply"})
    monkeypatch.setattr(graph, "bear_defends", lambda state: {"bear_defends": "reply"})
    monkeypatch.setattr(
        graph,
        "judge",
        lambda state: {
            "decision": "HOLD",
            "confidence": 0.5,
            "summary": "uncertain",
            "risks": [],
            "limitations": [],
        },
    )
    result = graph.build_graph().invoke({"raw_ticker": "AAPL", "constraint": "long term"})
    assert result["decision"] == "HOLD"
    assert result["confidence"] == 0.5
