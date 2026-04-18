import os
import re
import sys
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import ollama
import requests
import yfinance as yf

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
UA = {"User-Agent": "Mozilla/5.0"}
POS = ("beat", "bull", "growth", "gain", "rally", "up", "strong", "surge", "profit")
NEG = ("miss", "bear", "loss", "risk", "fall", "drop", "weak", "lawsuit", "cut")


def chat(prompt):
    return ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])["message"]["content"].strip()


def read_input():
    if len(sys.argv) >= 2:
        ticker = sys.argv[1].strip()
        if len(sys.argv) >= 3:
            return ticker, " ".join(sys.argv[2:]).strip()
        return ticker, input("Constraint: ").strip()
    ticker = input("Ticker: ").strip()
    constraint = input("Constraint: ").strip()
    return ticker, constraint


def pick_symbol(raw):
    raw = raw.upper().replace(" ", "")
    candidates = [raw]
    if "-" not in raw and "." not in raw:
        candidates.append(f"{raw}-USD")
    for symbol in candidates:
        try:
            if not yf.Ticker(symbol).history(period="2d").empty:
                return symbol
        except Exception:
            pass
    return raw


def market_data(symbol):
    try:
        df = yf.Ticker(symbol).history(period="7d", interval="1d")
    except Exception:
        return "No price data."
    if df.empty:
        return "No price data."
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
    move = (last - prev) / prev * 100 if prev else 0
    low = float(df["Low"].tail(5).min())
    high = float(df["High"].tail(5).max())
    vol = int(df["Volume"].iloc[-1]) if "Volume" in df else 0
    return f"last close {last:.4f}, 1d change {move:+.2f}%, 5d range {low:.4f}-{high:.4f}, volume {vol}"


def news_headlines(symbol):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(symbol)}&region=US&lang=en-US"
    try:
        xml = requests.get(url, headers=UA, timeout=8).text
        root = ET.fromstring(xml)
        items = [i.findtext("title", "").strip() for i in root.findall(".//item")]
        items = [x for x in items if x]
        if items:
            return items[:5]
    except Exception:
        pass
    try:
        items = yf.Ticker(symbol).news[:5]
        return [x["title"] for x in items if x.get("title")]
    except Exception:
        return []


def sentiment(headlines):
    score = 0
    for headline in headlines:
        text = headline.lower()
        score += sum(word in text for word in POS)
        score -= sum(word in text for word in NEG)
    return score


def ask_agents(ticker, constraint):
    symbol = pick_symbol(ticker)
    price = market_data(symbol)
    headlines = news_headlines(symbol)
    news = "; ".join(headlines) if headlines else "No recent headlines found."
    mood = sentiment(headlines)
    base = f"Ticker: {ticker}\nSymbol used: {symbol}\nConstraint: {constraint}\nPrice: {price}\nNews sentiment score: {mood}\nNews: {news}"

    bull = chat(
        base
        + "\nYou are Bull. Argue for BUY in 3 short bullets. Tie your view to the constraint and the market data."
    )
    bear = chat(
        base
        + "\nBull argument:\n"
        + bull
        + "\nYou are Bear. Directly attack Bull's points and make the strongest SELL case in 3 short bullets."
    )
    defense = chat(
        base
        + "\nBull argument:\n"
        + bull
        + "\nBear attack:\n"
        + bear
        + "\nYou are Bull again. Defend Bull against Bear in 3 short bullets."
    )
    judge = chat(
        base
        + "\nBull:\n"
        + bull
        + "\nBear:\n"
        + bear
        + "\nBull defense:\n"
        + defense
        + "\nYou are Judge. Return exactly:\nDecision: BUY/SELL/NEUTRAL\nConfidence: 0 to 1\nReasoning: short reason"
    )
    return symbol, bull, bear, defense, judge


def main():
    ticker, constraint = read_input()
    symbol, bull, bear, defense, judge = ask_agents(ticker, constraint)
    print(f"\nTicker: {ticker} -> {symbol}")
    print("\nBull:\n" + bull)
    print("\nBear:\n" + bear)
    print("\nBull defense:\n" + defense)
    print("\nJudge:\n" + judge)
    m = re.search(r"Decision:\s*(BUY|SELL|NEUTRAL).*?Confidence:\s*([0-9]*\.?[0-9]+).*?Reasoning:\s*(.*)", judge, re.I | re.S)
    if m:
        print(f"\nFinal Decision: {m.group(1).upper()}")
        print(f"Confidence: {m.group(2)}")
        print(f"Reasoning: {m.group(3).strip()}")


if __name__ == "__main__":
    main()
