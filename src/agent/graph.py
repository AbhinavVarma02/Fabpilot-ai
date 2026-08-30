"""LangGraph assembly for FabPilot AI."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    confidence_check_node,
    human_review_node,
    shap_explanation_node,
    summary_generation_node,
    wafer_defect_classification_node,
    yield_prediction_node,
)
from src.agent.state import FabPilotState


def build_fabpilot_graph():
    """Build and compile the FabPilot LangGraph workflow."""

    graph = StateGraph(FabPilotState)
    graph.add_node("yield_prediction", yield_prediction_node)
    graph.add_node("shap_explanation", shap_explanation_node)
    graph.add_node("wafer_defect_classification", wafer_defect_classification_node)
    graph.add_node("confidence_check", confidence_check_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("summary_generation", summary_generation_node)

    graph.add_edge(START, "yield_prediction")
    graph.add_edge("yield_prediction", "shap_explanation")
    graph.add_edge("shap_explanation", "wafer_defect_classification")
    graph.add_edge("wafer_defect_classification", "confidence_check")
    graph.add_edge("confidence_check", "human_review")
    graph.add_edge("human_review", "summary_generation")
    graph.add_edge("summary_generation", END)

    return graph.compile()
