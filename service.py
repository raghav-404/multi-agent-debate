from time import monotonic

import psycopg
from pydantic import ValidationError

from agents import EvidenceCitationError, JudgeOutputError
from config import get_settings
from graph import build_graph
from memory import save_run
from schemas import DebateRequest, DebateResponse, Evidence, JudgeDecision


class ConfigurationError(RuntimeError):
    pass


class PersistenceError(RuntimeError):
    pass


def run_debate(
    request: DebateRequest,
    *,
    graph_runner=None,
    evidence: list[Evidence] | None = None,
    persist: bool = True,
) -> DebateResponse:
    try:
        settings = get_settings()
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    if not settings.groq_api_key:
        raise ConfigurationError("GROQ_API_KEY is required to run a debate")

    started = monotonic()
    if graph_runner is None:
        graph_runner = build_graph()
    initial_state = {
        "raw_ticker": request.ticker,
        "constraint": request.constraint,
        "retry_count": 0,
        "model_calls": 0,
        "token_reports": [],
    }
    if evidence is not None:
        initial_state["provided_evidence"] = evidence
    result = graph_runner.invoke(initial_state)
    latency_ms = round((monotonic() - started) * 1000)
    try:
        decision = JudgeDecision.model_validate(
            {
                field: result.get(field)
                for field in (
                    "decision",
                    "confidence",
                    "summary",
                    "supporting_evidence",
                    "risks",
                    "limitations",
                )
            }
        )
    except ValidationError as exc:
        raise JudgeOutputError("Workflow returned an invalid decision") from exc
    cited_ids = set(decision.supporting_evidence)
    evidence_used = [item for item in result["evidence"] if item.id in cited_ids]
    if len({item.id for item in evidence_used}) != len(cited_ids):
        raise EvidenceCitationError("Judge cited evidence not present in this request")
    model_calls = result["model_calls"]
    reports = result["token_reports"]
    complete_usage = len(reports) == model_calls
    prompt_tokens = sum(item["prompt_tokens"] for item in reports) if complete_usage else None
    completion_tokens = (
        sum(item["completion_tokens"] for item in reports) if complete_usage else None
    )

    persisted = False
    if persist and settings.database_url:
        try:
            persisted = save_run(
                ticker=request.ticker,
                constraint=request.constraint,
                decision=decision.decision.value,
                confidence=decision.confidence,
                summary=decision.summary,
                risks=decision.risks,
                latency_ms=latency_ms,
                retry_count=result["retry_count"],
                model_name=settings.groq_model,
            )
        except psycopg.Error as exc:
            raise PersistenceError("Decision could not be stored in PostgreSQL") from exc
        if not persisted:
            raise PersistenceError("Decision could not be stored in PostgreSQL")

    return DebateResponse(
        ticker=request.ticker,
        constraint=request.constraint,
        decision=decision.decision,
        confidence=decision.confidence,
        summary=decision.summary,
        supporting_evidence=decision.supporting_evidence,
        evidence_used=evidence_used,
        risks=decision.risks,
        limitations=decision.limitations,
        latency_ms=latency_ms,
        retry_count=result["retry_count"],
        model_calls=model_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model_name=settings.groq_model,
        persisted=persisted,
    )
