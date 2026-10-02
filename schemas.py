import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DebateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str
    constraint: str = Field(min_length=1, max_length=500)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.^-]{0,14}", ticker):
            raise ValueError("ticker must be 1–15 letters, digits, dots, carets or hyphens")
        return ticker


class Decision(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^(price|news)_[1-9][0-9]*$")
    source: Literal["price", "news"]
    text: str = Field(min_length=1)


class JudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    decision: Decision
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=1, max_length=2000)
    supporting_evidence: list[str] = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
