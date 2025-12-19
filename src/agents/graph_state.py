import logging
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..llm_factory import LLMFactory
from ..models.graph_state import GraphState
from .category_routing import route_by_category
from .classifier import classifier_node
from .dialog import DialogAgent
from .erp_agent import erp_node
from .finalizer import finalizer_node
from .rag_agent import rag_node

logger = logging.getLogger(__name__)


def route_after_dialog(state: GraphState) -> str:
    if state.dialog_complete:
        return "classifier"
    return END


def build_unified_graph(
    llm_factory: LLMFactory,
    vector_store: Optional[Any] = None,
    erp_client: Optional[Any] = None,
) -> CompiledStateGraph:
    dialog_agent = DialogAgent(llm_factory)
    workflow = StateGraph(GraphState)

    workflow.add_node("dialog_agent", dialog_agent.agent_node)
    workflow.add_node("dialog_tools", dialog_agent.tool_node)
    workflow.add_node("dialog_structured", dialog_agent.structured_node)

    workflow.add_node("classifier", lambda state: classifier_node(state, llm_factory))

    workflow.add_node("rag", lambda state: rag_node(state, llm_factory, vector_store))
    workflow.add_node("erp", lambda state: erp_node(state, llm_factory, erp_client))
    workflow.add_node("finalizer", lambda state: finalizer_node(state, llm_factory))

    workflow.set_entry_point("dialog_agent")

    workflow.add_conditional_edges(
        "dialog_agent",
        dialog_agent.route_after_agent,
        {
            "tools": "dialog_tools",
            "structured": "dialog_structured",
            END: END,
        },
    )
    workflow.add_edge("dialog_tools", "dialog_agent")

    workflow.add_conditional_edges(
        "dialog_structured",
        route_after_dialog,
        {
            "classifier": "classifier",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "classifier",
        route_by_category,
        {
            "rag": "rag",
            "erp": "erp",
            "finalizer": "finalizer",
        },
    )

    workflow.add_edge("rag", END)
    workflow.add_edge("erp", END)
    workflow.add_edge("finalizer", END)

    compiled_graph = workflow.compile(checkpointer=InMemorySaver())
    return compiled_graph
