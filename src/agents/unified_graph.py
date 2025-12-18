import logging
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from ..llm_factory import LLMFactory
from ..models.graph_state import UnifiedGraphState
from .category_routing import route_by_category
from .classifier import classifier_node
from .dialog import DialogAgent
from .erp_agent import erp_node
from .finalizer import finalizer_node
from .rag_agent import rag_node

logger = logging.getLogger(__name__)


def route_after_dialog(state: UnifiedGraphState) -> str:
    """
    Маршрутизация после dialog_structured:
    - Если dialog_complete=True → classifier
    - Иначе → END (граф завершается, ждёт следующее сообщение пользователя)
    """
    if state.dialog_complete:
        return "classifier"
    return END


def build_unified_graph(
    llm_factory: LLMFactory,
    vector_store: Optional[Any] = None,
    erp_client: Optional[Any] = None,
):
    """
    Построить единый граф для всех агентов системы.

    Граф объединяет:
    - DialogAgent: сбор информации от пользователя
    - ClassifierAgent: определение категории проблемы
    - RAG Agent: поиск в базе знаний
    - ERP Agent: запросы к ERP-системе
    - Finalizer: формирование обращения для логистов

    Args:
        llm_factory: фабрика для создания LLM
        vector_store: векторное хранилище для RAG (опционально)
        erp_client: клиент для ERP API (опционально)

    Returns:
        Скомпилированный LangGraph
    """
    logger.info("Building unified multi-agent graph")

    # Создаем DialogAgent для доступа к его узлам
    dialog_agent = DialogAgent(llm_factory)

    # Создаем граф
    workflow = StateGraph(UnifiedGraphState)

    # === Добавление узлов DialogAgent ===
    workflow.add_node("dialog_agent", dialog_agent.agent_node)
    workflow.add_node("dialog_tools", dialog_agent.tool_node)
    workflow.add_node("dialog_structured", dialog_agent.structured_node)

    # === Добавление узла ClassifierAgent ===
    workflow.add_node("classifier", lambda state: classifier_node(state, llm_factory))

    # === Добавление специализированных агентов ===
    workflow.add_node("rag", lambda state: rag_node(state, llm_factory, vector_store))
    workflow.add_node("erp", lambda state: erp_node(state, llm_factory, erp_client))
    workflow.add_node("finalizer", lambda state: finalizer_node(state, llm_factory))

    # === Настройка edges ===
    workflow.set_entry_point("dialog_agent")

    # Внутренняя маршрутизация диалогового агента
    workflow.add_conditional_edges(
        "dialog_agent",
        dialog_agent.route_after_agent,
        {
            "tools": "dialog_tools",
            "structured": "dialog_structured",
            END: END,  # Агент сгенерировал ответ → завершаем граф, ждём пользователя
        },
    )
    workflow.add_edge("dialog_tools", "dialog_agent")

    # После извлечения данных → переход к классификатору
    workflow.add_conditional_edges(
        "dialog_structured",
        route_after_dialog,
        {
            "classifier": "classifier",
            END: END,
        },
    )

    # Classifier → маршрутизация по категориям
    workflow.add_conditional_edges(
        "classifier",
        route_by_category,
        {
            "rag": "rag",
            "erp": "erp",
            "finalizer": "finalizer",
        },
    )

    # Все специализированные агенты → END
    workflow.add_edge("rag", END)
    workflow.add_edge("erp", END)
    workflow.add_edge("finalizer", END)

    # Компиляция с checkpointer для сохранения состояния между сообщениями
    compiled_graph = workflow.compile(checkpointer=InMemorySaver())

    logger.info("Unified graph built successfully")

    return compiled_graph
