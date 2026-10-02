import sys
from time import monotonic

from pydantic import ValidationError

from agents import ModelError
from config import get_settings
from graph import build_graph
from market_data import MarketDataError, NewsDataError
from memory import init_db, save_run
from schemas import DebateRequest


def read_input() -> DebateRequest:
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Ticker: ")
    constraint = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Constraint: ")
    return DebateRequest(ticker=ticker, constraint=constraint)


def main() -> int:
    try:
        request = read_input()
        settings = get_settings()
        if not settings.groq_api_key:
            raise ModelError("GROQ_API_KEY is required to run a debate")
        started = monotonic()
        result = build_graph().invoke(
            {"raw_ticker": request.ticker, "constraint": request.constraint, "retry_count": 0}
        )
        latency_ms = round((monotonic() - started) * 1000)
        stored = False
        if settings.database_url:
            init_db()
            stored = save_run(
                ticker=result["ticker"],
                constraint=result["constraint"],
                decision=result["decision"],
                confidence=result["confidence"],
                summary=result["summary"],
                risks=result["risks"],
                latency_ms=latency_ms,
                retry_count=result["retry_count"],
                model_name=settings.groq_model,
            )
    except (ValidationError, ModelError, MarketDataError, NewsDataError, ValueError) as exc:
        print(f"Debate failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - the CLI must show an observable failure
        print(f"Debate failed unexpectedly: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Ticker: {result['ticker']}")
    print(f"Decision: {result['decision']} ({result['confidence']:.2f})")
    print(f"Summary: {result['summary']}")
    print("Evidence cited:")
    cited_ids = set(result["supporting_evidence"])
    for item in result["evidence"]:
        if item.id in cited_ids:
            print(f"  [{item.id}] {item.text}")
    print(f"Risks: {', '.join(result['risks']) or 'None listed'}")
    print(f"Limitations: {', '.join(result['limitations']) or 'None listed'}")
    print(f"Critique retries: {result['retry_count']}")
    print(f"Latency: {latency_ms} ms")
    print("Saved to PostgreSQL" if stored else "Not saved: DATABASE_URL is unset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
