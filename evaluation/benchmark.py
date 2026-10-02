import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from agents import EvidenceCitationError, JudgeOutputError, ModelError, collect_context
from config import get_settings
from evaluation.baseline import run_baseline
from market_data import MarketDataError, NewsDataError
from schemas import DebateRequest
from service import ConfigurationError, PersistenceError, run_debate

SCENARIOS_PATH = Path(__file__).with_name("scenarios.json")
RESULTS_DIR = Path(__file__).with_name("results")


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict]:
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Scenario file must contain a nonempty list")
    ids = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict) or not isinstance(scenario.get("id"), str):
            raise TypeError("Every scenario needs a string id")
        if scenario["id"] in ids:
            raise ValueError(f"Duplicate scenario id: {scenario['id']}")
        ids.add(scenario["id"])
        DebateRequest(ticker=scenario["ticker"], constraint=scenario["constraint"])
    return scenarios


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


def summarize(records: list[dict], scenario_ids: list[str], repeats: int) -> dict:
    summary = {}
    for method in ("baseline", "debate"):
        selected = [record for record in records if record["method"] == method]
        completed = [record for record in selected if record["status"] == "completed"]
        latencies = [record["latency_ms"] for record in completed]
        format_checked = [
            record for record in selected if record["structured_output_valid"] is not None
        ]
        citation_checked = [record for record in selected if record["citation_valid"] is not None]
        usage_reported = [
            record
            for record in completed
            if record["prompt_tokens"] is not None and record["completion_tokens"] is not None
        ]
        comparable = 0
        consistent = 0
        if repeats >= 2:
            for scenario_id in scenario_ids:
                decisions = [
                    record["decision"]
                    for record in completed
                    if record["scenario_id"] == scenario_id
                ]
                if len(decisions) == repeats:
                    comparable += 1
                    consistent += len(set(decisions)) == 1
        count = len(selected)
        summary[method] = {
            "attempts": count,
            "completed": len(completed),
            "failed": count - len(completed),
            "completion_rate": round(len(completed) / count, 4) if count else None,
            "failure_rate": round((count - len(completed)) / count, 4) if count else None,
            "structured_output_validity_rate": (
                round(
                    sum(record["structured_output_valid"] for record in format_checked)
                    / len(format_checked),
                    4,
                )
                if format_checked
                else None
            ),
            "citation_validity_rate": (
                round(
                    sum(record["citation_valid"] for record in citation_checked)
                    / len(citation_checked),
                    4,
                )
                if citation_checked
                else None
            ),
            "retry_rate": (
                round(sum(record["retry_count"] > 0 for record in completed) / len(completed), 4)
                if completed
                else None
            ),
            "latency_ms_mean": round(mean(latencies), 2) if latencies else None,
            "latency_ms_p50": percentile(latencies, 0.5),
            "latency_ms_p95": percentile(latencies, 0.95),
            "model_calls_completed_total": sum(record["model_calls"] for record in completed),
            "model_calls_completed_mean": (
                round(mean(record["model_calls"] for record in completed), 2) if completed else None
            ),
            "token_usage_reported_runs": len(usage_reported),
            "prompt_tokens_reported_total": (
                sum(record["prompt_tokens"] for record in usage_reported)
                if usage_reported
                else None
            ),
            "completion_tokens_reported_total": (
                sum(record["completion_tokens"] for record in usage_reported)
                if usage_reported
                else None
            ),
            "comparable_scenarios": comparable,
            "consistent_scenarios": consistent,
            "decision_consistency_rate": round(consistent / comparable, 4) if comparable else None,
        }
    return summary


def failure_record(scenario_id: str, method: str, repeat: int, exc: Exception) -> dict:
    if isinstance(exc, EvidenceCitationError):
        structured_valid, citation_valid = True, False
    elif isinstance(exc, JudgeOutputError):
        structured_valid, citation_valid = False, None
    else:
        structured_valid, citation_valid = None, None
    safe_error = isinstance(
        exc, (ModelError, MarketDataError, NewsDataError, ConfigurationError, PersistenceError)
    )
    return {
        "scenario_id": scenario_id,
        "method": method,
        "repeat": repeat,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc) if safe_error else None,
        "structured_output_valid": structured_valid,
        "citation_valid": citation_valid,
    }


def evaluate_scenarios(
    scenarios: list[dict],
    repeats: int,
    *,
    collect=collect_context,
    baseline=run_baseline,
    debate=run_debate,
) -> dict:
    if repeats < 2:
        raise ValueError("At least two repeats are required for consistency measurement")
    records = []
    contexts = []
    for scenario in scenarios:
        request = DebateRequest(ticker=scenario["ticker"], constraint=scenario["constraint"])
        scenario_id = scenario["id"]
        try:
            context = collect({"raw_ticker": request.ticker, "constraint": request.constraint})
            evidence = context["evidence"]
            contexts.append(
                {"scenario_id": scenario_id, "evidence": [item.model_dump() for item in evidence]}
            )
        except Exception as exc:  # noqa: BLE001 - record the failure and continue every scenario
            contexts.append({"scenario_id": scenario_id, "error_type": type(exc).__name__})
            for repeat in range(1, repeats + 1):
                for method in ("baseline", "debate"):
                    records.append(failure_record(scenario_id, method, repeat, exc))
            continue
        for repeat in range(1, repeats + 1):
            for method in ("baseline", "debate"):
                try:
                    result = (
                        baseline(request, evidence)
                        if method == "baseline"
                        else debate(request, evidence=evidence, persist=False)
                    )
                    records.append(
                        {
                            "scenario_id": scenario_id,
                            "method": method,
                            "repeat": repeat,
                            "status": "completed",
                            "structured_output_valid": True,
                            "citation_valid": True,
                            **result.model_dump(mode="json"),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - keep evaluating the remaining runs
                    records.append(failure_record(scenario_id, method, repeat, exc))
    return {
        "scenario_count": len(scenarios),
        "repeats": repeats,
        "contexts": contexts,
        "records": records,
        "metrics": summarize(records, [scenario["id"] for scenario in scenarios], repeats),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in live baseline versus debate evaluation")
    parser.add_argument("--live", action="store_true", help="allow real provider and Groq calls")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live to allow external API calls")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    try:
        settings = get_settings()
    except ValueError as exc:
        parser.error(str(exc))
    if not settings.groq_api_key:
        parser.error("GROQ_API_KEY is required for live evaluation")
    scenarios = load_scenarios()
    result = evaluate_scenarios(scenarios, args.repeats)
    result["generated_at_utc"] = datetime.now(UTC).isoformat()
    result["dataset"] = "evaluation/scenarios.json"
    result["model_name"] = settings.groq_model
    output = args.output or RESULTS_DIR / f"run-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved actual evaluation results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
