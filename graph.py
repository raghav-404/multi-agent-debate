from typing import Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import bear_critique, bull_analysis, bull_revision, collect_context, critic, judge
from config import get_settings
from schemas import Evidence


class State(TypedDict):
    raw_ticker: str
    constraint: str
    retry_count: int
    ticker: NotRequired[str]
    evidence: NotRequired[list[Evidence]]
    bull_argument: NotRequired[str]
    bear_critique: NotRequired[str]
    bull_revision: NotRequired[str]
    critic_feedback: NotRequired[str]
    decision: NotRequired[str]
    confidence: NotRequired[float]
    summary: NotRequired[str]
    supporting_evidence: NotRequired[list[str]]
    risks: NotRequired[list[str]]
    limitations: NotRequired[list[str]]


def confidence_router(state: State) -> Literal["critic", "done"]:
    if state["confidence"] < get_settings().retry_threshold and state["retry_count"] < 1:
        return "critic"
    return "done"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("collect_context", collect_context)
    graph.add_node("bull_analysis", bull_analysis)
    graph.add_node("bear_critique", bear_critique)
    graph.add_node("bull_revision", bull_revision)
    graph.add_node("judge", judge)
    graph.add_node("critic", critic)
    graph.add_edge(START, "collect_context")
    graph.add_edge("collect_context", "bull_analysis")
    graph.add_edge("bull_analysis", "bear_critique")
    graph.add_edge("bear_critique", "bull_revision")
    graph.add_edge("bull_revision", "judge")
    graph.add_conditional_edges("judge", confidence_router, {"critic": "critic", "done": END})
    graph.add_edge("critic", "judge")
    return graph.compile()
