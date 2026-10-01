from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents import bear_attack, bear_defends, bull, bull_defense, judge, user_input


class State(TypedDict, total=False):
    raw_ticker: str
    ticker: str
    constraint: str
    market_data: str
    news: list[str]
    bull_argument: str
    bear_attack: str
    bull_defense: str
    bear_defends: str
    decision: str
    confidence: float
    summary: str
    risks: list[str]
    limitations: list[str]


def build_graph():
    graph = StateGraph(State)
    graph.add_node("user_input", user_input)
    graph.add_node("bull", bull)
    graph.add_node("bear_attack", bear_attack)
    graph.add_node("bull_defense", bull_defense)
    graph.add_node("bear_defends", bear_defends)
    graph.add_node("judge", judge)
    graph.add_edge(START, "user_input")
    graph.add_edge("user_input", "bull")
    graph.add_edge("bull", "bear_attack")
    graph.add_edge("bear_attack", "bull_defense")
    graph.add_edge("bull_defense", "bear_defends")
    graph.add_edge("bear_defends", "judge")
    graph.add_edge("judge", END)
    return graph.compile()
