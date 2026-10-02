# Reliable Multi-Agent Financial Debate System

A learning project that asks role-specialized agents using one configured Groq model to debate a ticker, then asks a Judge for a validated decision. It is a decision-support demonstration, not a price predictor, trading bot, or financial advice.

## Current status: Phase 2

The CLI runs this LangGraph flow:

```text
collect context → Bull → Bear → Bull revision → Judge
                                            ↓
                              confidence below threshold?
                                yes → Critic → Judge → end
                                 no ────────────────→ end
```

Retrieved price and headline items receive request-level IDs such as `price_1` and `news_1`. The Judge returns JSON validated by Pydantic and must cite at least one ID from that request. A confidence below the configured threshold triggers one Critic pass and a final Judge revision. Invalid output and external failures stop the run. PostgreSQL storage is optional; if configured, a database failure fails the CLI run.

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
```

`GROQ_MODEL` defaults to `openai/gpt-oss-20b`. `RETRY_THRESHOLD` defaults to `0.6`. `DATABASE_URL` is optional. If you enable it, create the PostgreSQL database named in your URL before running the CLI; the program creates its table. No pgvector extension is needed.

## Checks

```bash
uv run pytest -q
uv run ruff check .
```

Tests use mocks and make no live model, market, news, or database calls. There are no benchmark results yet.

## Limitations

The current price and headline providers may omit data or fail. Headlines are not fact checked; their IDs show which provider items were used, not that those items are true. An invalid Judge response fails instead of receiving another model attempt. There is no API or evaluation harness yet. Do not use the output as investment advice.
