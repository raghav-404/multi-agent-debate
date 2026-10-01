import xml.etree.ElementTree as ET
from math import isfinite
from urllib.parse import quote_plus

import requests
import yfinance as yf


class MarketDataError(RuntimeError):
    pass


class NewsDataError(RuntimeError):
    pass


def get_market_data(ticker: str) -> str:
    try:
        frame = yf.Ticker(ticker).history(period="7d", interval="1d", timeout=8)
        if frame.empty:
            raise MarketDataError(f"No recent price data for {ticker}")
        last = float(frame["Close"].iloc[-1])
        previous = float(frame["Close"].iloc[-2]) if len(frame) > 1 else last
        low = float(frame["Low"].tail(5).min())
        high = float(frame["High"].tail(5).max())
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(f"Price provider failed for {ticker}") from exc
    if not all(isfinite(value) and value > 0 for value in (last, previous, low, high)):
        raise MarketDataError(f"Price provider returned invalid numbers for {ticker}")
    change = (last - previous) / previous * 100
    return f"last close {last:.4f}; 1d change {change:+.2f}%; 5d range {low:.4f}-{high:.4f}"


def get_headlines(ticker: str) -> list[str]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(ticker)}&region=US&lang=en-US"
    try:
        response = requests.get(url, headers={"User-Agent": "financial-debate-demo/1.0"}, timeout=8)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as exc:
        raise NewsDataError(f"News provider failed for {ticker}") from exc
    return [
        title for item in root.findall(".//item") if (title := item.findtext("title", "").strip())
    ][:5]
