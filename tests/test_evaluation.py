import sys
from types import SimpleNamespace

import pytest

from agents import EvidenceCitationError, JudgeOutputError, ModelReply
from evaluation import baseline, benchmark
from schemas import DebateResponse, Evidence


def example_response(request, evidence, *, method):
    return DebateResponse(
        ticker=request.ticker,
        constraint=request.constraint,
        decision="HOLD",
        confidence=0.5,
        summary="uncertain",
        supporting_evidence=["price_1"],
        evidence_used=evidence,
        risks=[],
        limitations=[],
        latency_ms=100 if method == "baseline" else 200,
        retry_count=0,
        model_calls=1 if method == "baseline" else 4,
        prompt_tokens=10 if method == "baseline" else None,
        completion_tokens=5 if method == "baseline" else None,
        model_name="test-model",
        persisted=False,
    )


def test_fixed_dataset_has_18_unique_valid_scenarios():
    scenarios = benchmark.load_scenarios()
    assert len(scenarios) == 18
    assert len({scenario["id"] for scenario in scenarios}) == 18


def test_baseline_uses_one_model_call_and_validates_citation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        baseline, "get_settings", lambda: SimpleNamespace(groq_api_key="test", groq_model="model")
    )

    def fake_chat(prompt, *, structured=False):
        calls.append((prompt, structured))
        return ModelReply(
            '{"decision":"HOLD","confidence":0.5,"summary":"uncertain",'
            '"supporting_evidence":["price_1"],"risks":[],"limitations":[]}',
            prompt_tokens=10,
            completion_tokens=5,
        )

    monkeypatch.setattr(baseline, "chat", fake_chat)
    from schemas import DebateRequest

    result = baseline.run_baseline(
        DebateRequest(ticker="AAPL", constraint="long term"),
        [Evidence(id="price_1", source="price", text="price summary")],
    )
    assert len(calls) == 1
    assert calls[0][1] is True
    assert result.model_calls == 1
    assert result.prompt_tokens == 10
    assert result.persisted is False


def test_baseline_rejects_invented_evidence(monkeypatch):
    monkeypatch.setattr(
        baseline, "get_settings", lambda: SimpleNamespace(groq_api_key="test", groq_model="model")
    )
    monkeypatch.setattr(
        baseline,
        "chat",
        lambda *args, **kwargs: ModelReply(
            '{"decision":"HOLD","confidence":0.5,"summary":"uncertain",'
            '"supporting_evidence":["news_9"],"risks":[],"limitations":[]}'
        ),
    )
    from schemas import DebateRequest

    with pytest.raises(EvidenceCitationError):
        baseline.run_baseline(
            DebateRequest(ticker="AAPL", constraint="long term"),
            [Evidence(id="price_1", source="price", text="price summary")],
        )


def test_full_dataset_is_processed_with_offline_fakes():
    scenarios = benchmark.load_scenarios()
    calls = {"collect": 0, "baseline": 0, "debate": 0}

    def collect(state):
        calls["collect"] += 1
        return {"evidence": [Evidence(id="price_1", source="price", text="price summary")]}

    def run_baseline(request, evidence):
        calls["baseline"] += 1
        return example_response(request, evidence, method="baseline")

    def run_debate(request, *, evidence, persist):
        assert persist is False
        calls["debate"] += 1
        return example_response(request, evidence, method="debate")

    result = benchmark.evaluate_scenarios(
        scenarios, 2, collect=collect, baseline=run_baseline, debate=run_debate
    )
    assert calls == {"collect": 18, "baseline": 36, "debate": 36}
    assert len(result["records"]) == 72
    assert result["metrics"]["baseline"]["completion_rate"] == 1
    assert result["metrics"]["debate"]["decision_consistency_rate"] == 1


def test_scenario_failure_does_not_stop_later_scenarios():
    scenarios = [
        {"id": "a", "ticker": "AAPL", "constraint": "long term"},
        {"id": "b", "ticker": "MSFT", "constraint": "long term"},
    ]

    def collect(state):
        if state["raw_ticker"] == "AAPL":
            raise RuntimeError("offline")
        return {"evidence": [Evidence(id="price_1", source="price", text="price summary")]}

    result = benchmark.evaluate_scenarios(
        scenarios,
        2,
        collect=collect,
        baseline=lambda request, evidence: example_response(request, evidence, method="baseline"),
        debate=lambda request, *, evidence, persist: example_response(
            request, evidence, method="debate"
        ),
    )
    assert len(result["records"]) == 8
    assert sum(record["status"] == "failed" for record in result["records"]) == 4
    assert sum(record["status"] == "completed" for record in result["records"]) == 4
    assert result["contexts"][1]["scenario_id"] == "b"


def test_metric_calculation_and_failure_categories():
    records = [
        {
            "scenario_id": "a",
            "method": "baseline",
            "repeat": repeat,
            "status": "completed",
            "structured_output_valid": True,
            "citation_valid": True,
            "decision": "HOLD",
            "latency_ms": latency,
            "retry_count": 0,
            "model_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
        }
        for repeat, latency in [(1, 100), (2, 200)]
    ]
    records.extend(
        [
            benchmark.failure_record("a", "debate", 1, JudgeOutputError("invalid")),
            benchmark.failure_record("a", "debate", 2, EvidenceCitationError("unknown")),
        ]
    )
    metrics = benchmark.summarize(records, ["a"], 2)
    assert metrics["baseline"]["latency_ms_mean"] == 150
    assert metrics["baseline"]["latency_ms_p95"] == 195
    assert metrics["baseline"]["model_calls_completed_total"] == 2
    assert metrics["baseline"]["prompt_tokens_reported_total"] == 20
    assert metrics["baseline"]["decision_consistency_rate"] == 1
    assert metrics["debate"]["completion_rate"] == 0
    assert metrics["debate"]["structured_output_validity_rate"] == 0.5
    assert metrics["debate"]["citation_validity_rate"] == 0


def test_live_flag_is_required_before_any_provider_call(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
