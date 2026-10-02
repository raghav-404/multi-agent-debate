import sys

import psycopg
from psycopg.types.json import Jsonb

from config import get_settings


def init_db() -> bool:
    database_url = get_settings().database_url
    if not database_url:
        return False
    with (
        psycopg.connect(database_url, connect_timeout=5) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS debate_decisions (
                    id BIGSERIAL PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    trade_constraint TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                    summary TEXT NOT NULL,
                    risks JSONB NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    retry_count INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
        )
    return True


def save_run(
    *,
    ticker: str,
    constraint: str,
    decision: str,
    confidence: float,
    summary: str,
    risks: list[str],
    latency_ms: int,
    retry_count: int,
    model_name: str,
) -> bool:
    database_url = get_settings().database_url
    if not database_url:
        return False
    with (
        psycopg.connect(database_url, connect_timeout=5) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            """
                INSERT INTO debate_decisions
                    (ticker, trade_constraint, decision, confidence, summary, risks,
                     latency_ms, retry_count, model_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
            (
                ticker,
                constraint,
                decision,
                confidence,
                summary,
                Jsonb(risks),
                latency_ms,
                retry_count,
                model_name,
            ),
        )
    return True


def main() -> int:
    try:
        if not init_db():
            print("DATABASE_URL is required to initialize PostgreSQL", file=sys.stderr)
            return 1
    except (psycopg.Error, ValueError) as exc:
        print(f"PostgreSQL initialization failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print("PostgreSQL table debate_decisions is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
