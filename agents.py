import asyncio
import re
import xml.etree.ElementTree as ET
import traceback
from urllib.parse import quote_plus

import ollama
import requests
import yfinance as yf

from config import EVAL_EMBED_MODEL, EVAL_LLM_MODEL, MODEL, NEG_WORDS, NEWS_LIMIT, POS_WORDS, PRICE_INTERVAL, PRICE_PERIOD, RETRY_THRESHOLD, USER_AGENT
from memory import load_last

UA = {"User-Agent": USER_AGENT}


def chat(prompt):
    return ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])["message"]["content"].strip()


def pick_symbol(raw):
    raw = raw.upper().replace(" ", "")
    for symbol in (raw, f"{raw}-USD"):
        try:
            if not yf.Ticker(symbol).history(period="2d").empty:
                return symbol
        except Exception:
            pass
    return raw


def market_data(symbol):
    try:
        df = yf.Ticker(symbol).history(period=PRICE_PERIOD, interval=PRICE_INTERVAL)
    except Exception:
        return "No price data."
    if df.empty:
        return "No price data."
    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
    move = (last - prev) / prev * 100 if prev else 0
    low = float(df["Low"].tail(5).min())
    high = float(df["High"].tail(5).max())
    return f"last close {last:.4f}, 1d change {move:+.2f}%, 5d range {low:.4f}-{high:.4f}"


def news_headlines(symbol):
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(symbol)}&region=US&lang=en-US"
    try:
        root = ET.fromstring(requests.get(url, headers=UA, timeout=8).text)
        items = [i.findtext("title", "").strip() for i in root.findall(".//item")]
        items = [x for x in items if x]
        if items:
            return items[:NEWS_LIMIT]
    except Exception:
        pass
    try:
        return [x["title"] for x in yf.Ticker(symbol).news[:NEWS_LIMIT] if x.get("title")]
    except Exception:
        return []


def sentiment(headlines):
    score = 0
    for headline in headlines:
        text = headline.lower()
        score += sum(word in text for word in POS_WORDS)
        score -= sum(word in text for word in NEG_WORDS)
    return score


def prompt(state):
    news = "; ".join(state.get("news", [])) or "No recent headlines."
    history = "\n".join(state.get("history", [])) or "No debate yet."
    memory = (
        f"Previous decision: {state['previous_decision']} ({state['previous_confidence']:.2f})\n"
        if state.get("previous_decision")
        else ""
    )
    return (
        f"Ticker: {state['ticker']}\n"
        f"Symbol: {state['symbol']}\n"
        f"Constraint: {state['constraint']}\n"
        f"Market: {state['market_data']}\n"
        f"News sentiment: {state['news_sentiment']}\n"
        f"News: {news}\n"
        f"{memory}"
        f"History:\n{history}"
    )


def user_input(state):
    symbol = pick_symbol(state["raw_ticker"])
    news = news_headlines(symbol)
    prev = load_last(symbol) or {}
    return {
        "ticker": state["raw_ticker"],
        "symbol": symbol,
        "market_data": market_data(symbol),
        "news": news,
        "news_sentiment": sentiment(news),
        **prev,
        "history": [f"User: {state['raw_ticker']} | Constraint: {state['constraint']} | Symbol: {symbol}"],
    }


def bull(state):
    text = chat(prompt(state) + "\nYou are Bull. Argue for BUY in 3 short bullets.")
    return {"bull_argument": text, "history": [f"Bull:\n{text}"]}


def bear_attack(state):
    text = chat(prompt(state) + "\nBull argument:\n" + state["bull_argument"] + "\nYou are Bear. Directly attack Bull and argue SELL in 3 short bullets.")
    return {"bear_attack": text, "history": [f"Bear attack:\n{text}"]}


def bull_defense(state):
    text = chat(
        prompt(state)
        + "\nBull argument:\n"
        + state["bull_argument"]
        + "\nBear attack:\n"
        + state["bear_attack"]
        + "\nYou are Bull again. Defend the BUY case in 3 short bullets."
    )
    return {"bull_defense": text, "history": [f"Bull defense:\n{text}"]}


def bear_defends(state):
    text = chat(
        prompt(state)
        + "\nBull argument:\n"
        + state["bull_argument"]
        + "\nBear attack:\n"
        + state["bear_attack"]
        + "\nBull defense:\n"
        + state["bull_defense"]
        + "\nYou are Bear again. Respond to Bull's defense and keep the SELL case strong in 3 short bullets."
    )
    return {"bear_defends": text, "history": [f"Bear defends:\n{text}"]}


def judge(state):
    text = chat(
        prompt(state)
        + "\nBull:\n"
        + state["bull_argument"]
        + "\nBear attack:\n"
        + state["bear_attack"]
        + "\nBull defense:\n"
        + state["bull_defense"]
        + "\nBear defends:\n"
        + state["bear_defends"]
        + "\nYou are Judge. Return exactly:\nDecision: BUY/SELL/NEUTRAL\nConfidence: 0 to 1\nReasoning: short reason"
    )
    m = re.search(r"Decision:\s*(BUY|SELL|NEUTRAL).*?Confidence:\s*([0-9]*\.?[0-9]+).*?Reasoning:\s*(.*)", text, re.I | re.S)
    if m:
        return {
            "decision": m.group(1).upper(),
            "confidence": float(m.group(2)),
            "reasoning": m.group(3).strip(),
            "weak": float(m.group(2)) < RETRY_THRESHOLD,
            "history": [f"Judge:\n{text}"],
        }
    return {"decision": "NEUTRAL", "confidence": 0.0, "reasoning": text, "weak": True, "history": [f"Judge:\n{text}"]}


def evaluate_reasoning(state):
    try:
        try:
            from ragas import SingleTurnSample
        except Exception:
            from ragas.dataset_schema import SingleTurnSample
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy
        from ragas.embeddings import HuggingFaceEmbeddings
        from openai import AsyncOpenAI
    except Exception:
        return {"eval_error": traceback.format_exc()}

    debate = "\n".join(state.get("history", []))
    sample = SingleTurnSample(
        user_input=f"{state['ticker']} | {state['constraint']}",
        response=debate + "\nFinal reasoning: " + state["reasoning"],
        retrieved_contexts=[
            state["market_data"],
            state["bull_argument"],
            state["bear_attack"],
            state["bull_defense"],
            state["bear_defends"],
        ],
    )
    client = AsyncOpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
    llm = llm_factory(EVAL_LLM_MODEL, provider="openai", client=client)
    emb = HuggingFaceEmbeddings(model=EVAL_EMBED_MODEL)

    async def score():
        return float(
            await AnswerRelevancy(llm=llm, embeddings=emb).ascore(
                user_input=sample.user_input,
                response=sample.response,
            )
        )

    try:
        return {"eval_relevancy": asyncio.run(score())}
    except Exception:
        return {"eval_error": traceback.format_exc()}
