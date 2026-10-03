# Reliable Multi-Agent Financial Debate System

A LangGraph-based financial debate system that asks role-specialized agents using one configured Groq model to debate a ticker, then asks a Judge for a validated decision. It is a decision-support demonstration, not a price predictor, trading bot, or financial advice.

## How it works

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

The stack is Python, LangGraph, Groq, FastAPI, Pydantic, PostgreSQL with psycopg, uv, pytest, and Ruff. Docker Compose runs only optional local PostgreSQL.

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
  "model_calls": 4,
  "prompt_tokens": null,
  "completion_tokens": null,
  "model_name": "openai/gpt-oss-20b",
  "persisted": false
}
```

`latency_ms` measures graph execution, excluding the optional database write. Token fields are `null` unless the provider reported usage for every model call. Invalid requests return HTTP 422; model or data provider failures return 502; configured storage failures return 503.

## Checks

```bash
uv run pytest -q
uv run ruff check .
```

Tests use mocks and make no live model, market, news, or database calls.

## Evaluation

`evaluation/scenarios.json` fixes 18 ticker/constraint scenarios without correctness labels. For each scenario, the harness retrieves context once and gives the same snapshot to a one-call baseline and the debate across repeated runs. Run the full dataset only when you intend to make live provider and Groq calls:

```bash
uv run python -m evaluation.benchmark --live --repeats 2
```

If all attempts complete, this makes 180–252 model calls. The command saves actual contexts, per-run records, and metrics under ignored `evaluation/results/`. There are no benchmark results in this repository. An example of the **output schema**, with no invented measurements, is:

```json
{
  "scenario_count": "integer",
  "repeats": "integer",
  "contexts": "one retrieved snapshot or failure per scenario",
  "records": "one completed or failed record per scenario, method, and repeat",
  "metrics": {
    "baseline": "completion, schema/citation validity, retry, latency, calls, consistency, reported tokens",
    "debate": "the same metrics"
  }
}
```

Metrics describe reliability and cost, not financial accuracy. Rates exclude cases where the property could not be evaluated; latency and model-call totals cover completed runs. Decision consistency compares scenarios with all repeats completed. Token totals cover only completed runs with usage reported for every call.

## Limitations

The price and headline providers may omit data or fail. Headlines are not fact checked; IDs show which provider items were used, not that those items are true. The fixed scenarios are reproducible, but live prices, headlines, and model responses change over time; saved context snapshots make each result inspectable. The dataset has no defensible decision labels, so the evaluation does not measure financial correctness. Do not use output as investment advice.
