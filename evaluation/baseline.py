from time import monotonic

from agents import chat, parse_judge_output, prompt
from config import get_settings
from schemas import DebateRequest, DebateResponse, Evidence
from service import ConfigurationError


def run_baseline(request: DebateRequest, evidence: list[Evidence]) -> DebateResponse:
    settings = get_settings()
    if not settings.groq_api_key:
        raise ConfigurationError("GROQ_API_KEY is required to run the baseline")
    if not evidence:
        raise ValueError("Baseline requires at least one evidence item")

    state = {"ticker": request.ticker, "constraint": request.constraint, "evidence": evidence}
    started = monotonic()
    reply = chat(
        prompt(state) + "\nYou are one impartial financial decision-support analyst. "
        "Return a JSON object with exactly: decision (BUY, HOLD or SELL), confidence "
        "(number 0 to 1), summary (short text), supporting_evidence (one or more "
        "retrieved evidence IDs), risks (array of strings), limitations (array of strings). "
        "Use only the listed evidence IDs; explain uncertainty. Do not invent facts.",
        structured=True,
    )
    decision = parse_judge_output(reply.content, evidence)
    latency_ms = round((monotonic() - started) * 1000)
    cited_ids = set(decision.supporting_evidence)
    return DebateResponse(
        ticker=request.ticker,
        constraint=request.constraint,
        decision=decision.decision,
        confidence=decision.confidence,
        summary=decision.summary,
        supporting_evidence=decision.supporting_evidence,
        evidence_used=[item for item in evidence if item.id in cited_ids],
        risks=decision.risks,
        limitations=decision.limitations,
        latency_ms=latency_ms,
        retry_count=0,
        model_calls=1,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        model_name=settings.groq_model,
        persisted=False,
    )
