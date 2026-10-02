import logging

from fastapi import FastAPI, HTTPException

from agents import ModelError
from market_data import MarketDataError, NewsDataError
from schemas import DebateRequest, DebateResponse
from service import ConfigurationError, PersistenceError, run_debate

logger = logging.getLogger(__name__)
app = FastAPI(title="Reliable Multi-Agent Financial Debate System")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/debates", response_model=DebateResponse)
def create_debate(request: DebateRequest) -> DebateResponse:
    try:
        return run_debate(request)
    except ConfigurationError as exc:
        logger.error("Debate configuration failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (MarketDataError, NewsDataError, ModelError) as exc:
        logger.warning("Debate provider or model failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except PersistenceError as exc:
        logger.error("Debate persistence failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected debate failure: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Unexpected internal failure") from exc
