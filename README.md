# Reliable Multi-Agent Financial Debate System

A learning project that asks role-specialized agents using one configured Groq model to debate a ticker, then asks a Judge for a validated decision. It is a decision-support demonstration, not a price predictor, trading bot, or financial advice.

## Current status: Phase 1

The CLI runs a linear LangGraph flow: collect recent price data and headlines → Bull → Bear → Bull response → Bear response → Judge. The Judge must return valid JSON that passes a Pydantic schema. Invalid output and external failures stop the run. PostgreSQL storage is optional; if `DATABASE_URL` is set, a successful decision is stored. A database failure fails the CLI run.

Confidence routing, evidence IDs, the API, and evaluation are planned for later phases. Agent arguments are interpretations, not verified market facts. One shared model plays all roles.

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

`GROQ_MODEL` defaults to `openai/gpt-oss-20b`. `RETRY_THRESHOLD` is reserved for Phase 2. `DATABASE_URL` is optional. If you enable it, create the PostgreSQL database named in your URL before running the CLI; the program creates its table. No pgvector extension is needed.

## Checks

```bash
uv run pytest -q
uv run ruff check .
```

Tests use mocks and make no live model, market, news, or database calls. There are no benchmark results yet.

## Limitations

The current price and headline providers may omit data or fail. Headlines are not fact checked. The Judge's confidence is a model estimate, not a calibrated probability. The workflow currently has no evidence-linked citations or retry path. Do not use its output as investment advice.
