import logging

from ..models.graph_state import GraphState

logger = logging.getLogger(__name__)

# Маппинг категорий на целевых агентов
CATEGORY_ROUTING: dict[str, str] = {
    "1": "finalizer",
    "2": "erp",
    "3": "rag",
    "4": "finalizer",
}

# Fallback агент для неизвестных категорий
DEFAULT_ROUTE = "finalizer"


def route_by_category(
    state: GraphState,
) -> str:
    category = state.category

    if category is None:
        return DEFAULT_ROUTE

    target = CATEGORY_ROUTING.get(category, DEFAULT_ROUTE)

    return target
