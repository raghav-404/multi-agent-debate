import pytest
import requests

import market_data


def test_news_provider_error_is_explicit(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout("timeout")

    monkeypatch.setattr(market_data.requests, "get", fail)
    with pytest.raises(market_data.NewsDataError, match="News provider failed"):
        market_data.get_headlines("AAPL")


def test_price_provider_error_is_explicit(monkeypatch):
    class BrokenTicker:
        def history(self, **kwargs):
            raise RuntimeError("offline")

    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: BrokenTicker())
    with pytest.raises(market_data.MarketDataError, match="Price provider failed"):
        market_data.get_market_data("AAPL")
