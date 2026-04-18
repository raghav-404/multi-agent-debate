import sys

from graph import build_graph
from memory import init_db, save_run


def read_input():
    if len(sys.argv) >= 2:
        ticker = sys.argv[1].strip()
        if len(sys.argv) >= 3:
            return ticker, " ".join(sys.argv[2:]).strip()
        return ticker, input("Constraint: ").strip()
    return input("Ticker: ").strip(), input("Constraint: ").strip()


def main():
    try:
        init_db()
    except Exception:
        pass
    ticker, constraint = read_input()
    out = build_graph().invoke({"raw_ticker": ticker, "constraint": constraint, "history": []})
    trend = (
        f"Last time: {out['previous_decision']} ({out['previous_confidence']:.2f}) -> "
        f"Now: {out['decision']} ({out['confidence']:.2f})"
        if out.get("previous_decision")
        else f"Now: {out['decision']} ({out['confidence']:.2f})"
    )
    print(f"\nTicker: {out['ticker']} -> {out['symbol']}")
    print(f"Constraint: {out['constraint']}")
    print(f"Trend: {trend}")
    print(f"\nBull:\n{out['bull_argument']}")
    print(f"\nBear attack:\n{out['bear_attack']}")
    print(f"\nBull defense:\n{out['bull_defense']}")
    print(f"\nBear defends:\n{out['bear_defends']}")
    print(f"\nFinal Decision: {out['decision']}")
    print(f"Confidence: {out['confidence']}")
    print(f"Reasoning: {out['reasoning']}")
    try:
        save_run(out["ticker"], out["constraint"], out["decision"], out["confidence"])
    except Exception:
        pass


if __name__ == "__main__":
    main()
