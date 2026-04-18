import operator
from typing_extensions import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import bear_attack, bear_defends, bull, bull_defense, judge, user_input


class State(TypedDict, total=False):
    raw_ticker: str
    ticker: str
    symbol: str
    constraint: str
    previous_decision: str
    previous_confidence: float
    market_data: str
    news: list[str]
    news_sentiment: int
    bull_argument: str
    bear_attack: str
    bull_defense: str
    bear_defends: str
    decision: str
    confidence: float
    reasoning: str
    weak: bool
    history: Annotated[list[str], operator.add]


def build_graph():
    g = StateGraph(State)
    g.add_node("user_input", user_input)
    g.add_node("bull", bull)
    g.add_node("bear_attack", bear_attack)
    g.add_node("bull_defense", bull_defense)
    g.add_node("bear_defends", bear_defends)
    g.add_node("judge", judge)
    g.add_edge(START, "user_input")
    g.add_edge("user_input", "bull")
    g.add_edge("bull", "bear_attack")
    g.add_edge("bear_attack", "bull_defense")
    g.add_edge("bull_defense", "bear_defends")
    g.add_edge("bear_defends", "judge")
    g.add_edge("judge", END)
    return g.compile()
