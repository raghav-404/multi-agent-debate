# Reliable Multi-Agent Financial Debate System

A learning project that asks role-specialized agents using one configured Groq model to debate a ticker, then asks a Judge for a validated decision. It is a decision-support demonstration, not a price predictor, trading bot, or financial advice.

## Current status: Phase 3

The CLI and API share this LangGraph flow:

```text
collect context → Bull → Bear → Bull revision → Judge
                                            ↓
                              confidence below threshold?
                                yes → Critic → Judge → end
                                 no ────────────────→ end
```

Retrieved price and headline items receive request-level IDs such as `price_1` and `news_1`. The Judge returns JSON validated by Pydantic and must cite at least one ID from that request. A confidence below the configured threshold triggers one Critic pass and a final Judge revision. Invalid output and external failures stop the request.

One shared model plays all roles. Agent arguments are interpretations, not verified market facts. The model's confidence controls routing but is not a calibrated probability.

## macOS setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked
cp .env.example .env
# Edit .env and set GROQ_API_KEY, then load it in your shell:
set -a
source .env
set +a
uv run python main.py AAPL "Long-term investing"
uv run uvicorn api:app --reload
```

The server listens at `http://127.0.0.1:8000`. `GET /health` reports only that the app is responding; it does not probe external services. `POST /debates` runs the workflow.

`GROQ_MODEL` defaults to `openai/gpt-oss-20b`. `RETRY_THRESHOLD` defaults to `0.6`. `DATABASE_URL` is optional. Without it, the response has `persisted: false`. If it is configured and storage fails, the request fails with HTTP 503; it does not return an unsaved decision.

## Optional PostgreSQL setup

Edit `.env` so `POSTGRES_PASSWORD` and the password in `DATABASE_URL` match, then run:

```bash
docker compose up --wait postgres
uv run python -m memory
```

Docker Compose runs PostgreSQL only. The second command creates the `debate_decisions` table. No pgvector extension is used. Run the schema command once before starting the API or CLI with `DATABASE_URL` set.

## API example

```bash
curl -X POST http://127.0.0.1:8000/debates \
  -H 'Content-Type: application/json' \
  -d '{"ticker":"AAPL","constraint":"Long-term investing"}'
```

Illustrative response shape, not a recorded result:

```json
{
  "decision": "HOLD",
  "confidence": 0.5,
  "summary": "Example summary only",
  "supporting_evidence": ["price_1"],
  "risks": ["Example risk"],
  "limitations": ["Example limitation"],
  "ticker": "AAPL",
  "constraint": "Long-term investing",
  "evidence_used": [{"id": "price_1", "source": "price", "text": "Example retrieved price summary"}],
  "latency_ms": 0,
  "retry_count": 0,
  "model_name": "openai/gpt-oss-20b",
  "persisted": false
}
```

`latency_ms` measures graph execution, excluding the optional database write. Invalid requests return HTTP 422; model or data provider failures return 502; configured storage failures return 503.

## Checks

```bash
uv run pytest -q
uv run ruff check .
```

Tests use mocks and make no live model, market, news, or database calls. There are no benchmark results yet.

## Limitations

The current price and headline providers may omit data or fail. Headlines are not fact checked; their IDs show which provider items were used, not that those items are true. An invalid Judge response fails instead of receiving another model attempt. There is no evaluation harness or measured decision quality yet. Do not use the output as investment advice.
