import sys

from graph import build_graph


def read_input():
    if len(sys.argv) >= 2:
        ticker = sys.argv[1].strip()
        if len(sys.argv) >= 3:
            return ticker, " ".join(sys.argv[2:]).strip()
        return ticker, input("Constraint: ").strip()
    return input("Ticker: ").strip(), input("Constraint: ").strip()


def main():
    ticker, constraint = read_input()
    out = build_graph().invoke({"raw_ticker": ticker, "constraint": constraint, "history": []})
    print(f"\nTicker: {out['ticker']} -> {out['symbol']}")
    print(f"Constraint: {out['constraint']}")
    print(f"\nBull:\n{out['bull_argument']}")
    print(f"\nBear attack:\n{out['bear_attack']}")
    print(f"\nBull defense:\n{out['bull_defense']}")
    print(f"\nBear defends:\n{out['bear_defends']}")
    print(f"\nFinal Decision: {out['decision']}")
    print(f"Confidence: {out['confidence']}")
    print(f"Reasoning: {out['reasoning']}")


if __name__ == "__main__":
    main()
