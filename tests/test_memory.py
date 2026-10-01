from types import SimpleNamespace

import pytest

import memory


class FakeConnection:
    def __init__(self):
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self

    def execute(self, query, params=None):
        self.executed.append((query, params))


def test_database_is_optional(monkeypatch):
    monkeypatch.setattr(memory, "get_settings", lambda: SimpleNamespace(database_url=None))
    assert memory.init_db() is False
    assert (
        memory.save_run(
            ticker="AAPL",
            constraint="long term",
            decision="HOLD",
            confidence=0.5,
            summary="uncertain",
            risks=[],
            latency_ms=100,
            retry_count=0,
            model_name="model",
        )
        is False
    )


def test_save_run_uses_parameters_and_no_vector(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr(
        memory, "get_settings", lambda: SimpleNamespace(database_url="postgresql://test")
    )
    monkeypatch.setattr(memory.psycopg, "connect", lambda *args, **kwargs: connection)
    memory.init_db()
    memory.save_run(
        ticker="AAPL",
        constraint="long term",
        decision="HOLD",
        confidence=0.5,
        summary="uncertain",
        risks=["volatility"],
        latency_ms=100,
        retry_count=0,
        model_name="model",
    )
    create_query, _ = connection.executed[0]
    insert_query, params = connection.executed[1]
    assert "vector" not in create_query.lower()
    assert "%s" in insert_query
    assert "long term" not in insert_query
    assert params[0] == "AAPL"
    assert params[1] == "long term"


def test_database_failure_propagates(monkeypatch):
    monkeypatch.setattr(
        memory, "get_settings", lambda: SimpleNamespace(database_url="postgresql://test")
    )

    def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(memory.psycopg, "connect", fail)
    with pytest.raises(RuntimeError, match="database unavailable"):
        memory.init_db()
