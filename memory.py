import psycopg2

from config import DATABASE_URL, MEMORY_TABLE, VECTOR_DIM


def enabled():
    return bool(DATABASE_URL)


def connect():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    if not enabled():
        return False
    with connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MEMORY_TABLE} (
                id bigserial PRIMARY KEY,
                ticker text NOT NULL,
                trade_constraint text NOT NULL,
                decision text NOT NULL,
                confidence real NOT NULL,
                embedding vector({VECTOR_DIM}) NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS {MEMORY_TABLE}_ticker_time_idx ON {MEMORY_TABLE} (ticker, created_at DESC)")
    return True


def load_last(ticker):
    if not enabled():
        return None
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT decision, confidence
                FROM {MEMORY_TABLE}
                WHERE ticker = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (ticker.upper(),),
            )
            row = cur.fetchone()
    except Exception:
        return None
    if not row:
        return None
    return {"previous_decision": row[0], "previous_confidence": float(row[1])}


def save_run(ticker, constraint, decision, confidence):
    if not enabled():
        return
    d = {"BUY": 1.0, "SELL": -1.0, "NEUTRAL": 0.0}.get(decision, 0.0)
    embedding = f"[{float(confidence):.4f},{d:.1f},{float(len(constraint)):.1f}]"
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {MEMORY_TABLE} (ticker, trade_constraint, decision, confidence, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            """,
            (ticker.upper(), constraint, decision, confidence, embedding),
        )
