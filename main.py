import sys

from pydantic import ValidationError

from agents import ModelError
from market_data import MarketDataError, NewsDataError
from schemas import DebateRequest
from service import ConfigurationError, PersistenceError, run_debate


def read_input() -> DebateRequest:
    ticker = sys.argv[1] if len(sys.argv) > 1 else input("Ticker: ")
    constraint = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Constraint: ")
    return DebateRequest(ticker=ticker, constraint=constraint)


def main() -> int:
    try:
        result = run_debate(read_input())
    except (
        ValidationError,
        ConfigurationError,
        ModelError,
        MarketDataError,
        NewsDataError,
        PersistenceError,
    ) as exc:
        print(f"Debate failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - the CLI must show an observable failure
        print(f"Debate failed unexpectedly: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Ticker: {result.ticker}")
    print(f"Decision: {result.decision} ({result.confidence:.2f})")
    print(f"Summary: {result.summary}")
    print("Evidence cited:")
    for item in result.evidence_used:
        print(f"  [{item.id}] {item.text}")
    print(f"Risks: {', '.join(result.risks) or 'None listed'}")
    print(f"Limitations: {', '.join(result.limitations) or 'None listed'}")
    print(f"Critique retries: {result.retry_count}")
    print(f"Debate latency: {result.latency_ms} ms")
    print("Saved to PostgreSQL" if result.persisted else "Not saved: DATABASE_URL is unset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
