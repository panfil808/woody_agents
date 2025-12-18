import logging
from typing import Literal

from ..models.graph_state import UnifiedGraphState

logger = logging.getLogger(__name__)

# Маппинг категорий на целевых агентов
CATEGORY_ROUTING: dict[int, str] = {
    1: "finalizer",
    2: "erp",
    3: "rag",
    4: "finalizer",
}

# Fallback агент для неизвестных категорий
DEFAULT_ROUTE = "finalizer"


def route_by_category(
    state: UnifiedGraphState,
) -> Literal["rag", "erp", "finalizer"]:
    """
    Маршрутизирует к соответствующему агенту на основе категории.

    Args:
        state: текущее состояние графа с определенной категорией

    Returns:
        Имя целевого узла: "rag", "erp" или "finalizer"
    """
    category = state.category

    if category is None:
        logger.warning("Категория не определена, маршрутизация на finalizer")
        return DEFAULT_ROUTE  # type: ignore

    target = CATEGORY_ROUTING.get(category, DEFAULT_ROUTE)

    logger.info(
        f"Маршрутизация: категория {category} ({state.category_name}) → {target}"
    )

    return target  # type: ignore