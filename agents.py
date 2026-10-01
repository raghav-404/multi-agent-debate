from groq import Groq
from pydantic import ValidationError

from config import get_settings
from market_data import get_headlines, get_market_data
from schemas import DebateRequest, JudgeDecision


class ModelError(RuntimeError):
    pass


class ModelOutputError(ModelError):
    pass


def chat(prompt: str, *, structured: bool = False) -> str:
    settings = get_settings()
    if not settings.groq_api_key:
        raise ModelError("GROQ_API_KEY is required to run a debate")
    try:
        client = Groq(api_key=settings.groq_api_key, timeout=20, max_retries=0)
        options = {"response_format": {"type": "json_object"}} if structured else {}
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            **options,
        )
    except Exception as exc:
        raise ModelError("Groq request failed; check your key, model and connection") from exc
    try:
        content = response.choices[0].message.content
    except (IndexError, AttributeError, TypeError) as exc:
        raise ModelOutputError("Groq returned a response without message content") from exc
    if not content or not content.strip():
        raise ModelOutputError("Groq returned an empty response")
    return content.strip()


def prompt(state: dict) -> str:
    headlines = "; ".join(state["news"]) or "No recent headlines returned."
    return (
        f"Ticker: {state['ticker']}\n"
        f"User constraint: {state['constraint']}\n"
        f"Retrieved price data: {state['market_data']}\n"
        f"Retrieved headlines: {headlines}\n"
        "Treat retrieved data as evidence; agent arguments are interpretations, not verified facts."
    )


def user_input(state: dict) -> dict:
    request = DebateRequest(ticker=state["raw_ticker"], constraint=state["constraint"])
    return {
        "ticker": request.ticker,
        "constraint": request.constraint,
        "market_data": get_market_data(request.ticker),
        "news": get_headlines(request.ticker),
    }


def bull(state: dict) -> dict:
    argument = chat(
        prompt(state) + "\nYou are Bull. Give three brief reasons for BUY. State uncertainty."
    )
    return {"bull_argument": argument}


def bear_attack(state: dict) -> dict:
    attack = chat(
        prompt(state) + f"\nBull argument: {state['bull_argument']}"
        "\nYou are Bear. Critique Bull and give three brief reasons for SELL. State uncertainty."
    )
    return {"bear_attack": attack}


def bull_defense(state: dict) -> dict:
    defense = chat(
        prompt(state) + f"\nBull argument: {state['bull_argument']}"
        f"\nBear critique: {state['bear_attack']}"
        "\nYou are Bull. Respond briefly to Bear's specific objections."
    )
    return {"bull_defense": defense}


def bear_defends(state: dict) -> dict:
    defense = chat(
        prompt(state) + f"\nBull defense: {state['bull_defense']}"
        "\nYou are Bear. Respond briefly, noting unresolved risks."
    )
    return {"bear_defends": defense}


def judge(state: dict) -> dict:
    raw = chat(
        prompt(state)
        + f"\nBull: {state['bull_argument']}"
        + f"\nBear critique: {state['bear_attack']}"
        + f"\nBull defense: {state['bull_defense']}"
        + f"\nBear response: {state['bear_defends']}"
        + "\nYou are the impartial Judge. Return a JSON object with exactly: "
        "decision (BUY, HOLD or SELL), confidence (number 0 to 1), summary (short text), "
        "risks (array of strings), limitations (array of strings). "
        "Do not treat agent arguments as verified evidence.",
        structured=True,
    )
    try:
        result = JudgeDecision.model_validate_json(raw)
    except ValidationError as exc:
        raise ModelOutputError("Judge returned an invalid decision") from exc
    return result.model_dump(mode="json")
