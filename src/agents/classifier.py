import logging

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from ..llm_factory import LLMFactory
from ..models.graph_state import GraphState
from .prompts import CLASSIFIER_AGENT_PROMPT

logger = logging.getLogger(__name__)


class CategoriesEnum:
    SEND_EARLIER = "Отправить доставку пораньше"
    UNASSIGN_DRIVER = "Снять водителя с доставки"
    CANCEL_DELIVERY = "Отменить доставку"
    DEMAND_DRIVER_INFO = "Запросить данные водителя для пропуска"


# Маппинг номеров категорий на их названия
VALID_CATEGORIES = {
    1: CategoriesEnum.SEND_EARLIER,
    2: CategoriesEnum.UNASSIGN_DRIVER,
    3: CategoriesEnum.CANCEL_DELIVERY,
    4: CategoriesEnum.DEMAND_DRIVER_INFO,
}


def classifier_node(
    state: GraphState,
    llm_factory: LLMFactory,
) -> dict:
    ttn_number = state.ttn_number
    problem_description = state.problem_description

    if not all([ttn_number, problem_description]):
        logger.warning("Неполная информация для классификации")
        error_msg = "✗ Не удалось классифицировать: недостаточно данных"
        return {
            "messages": [AIMessage(content=error_msg)],
        }

    llm = llm_factory.create_llm()

    collected_info = f"""
        Номер ТТН: {ttn_number}

        Описание проблемы:
        {problem_description}
    """

    messages = [
        SystemMessage(content=CLASSIFIER_AGENT_PROMPT),
        HumanMessage(content=collected_info),
    ]

    try:
        response = llm.invoke(messages)
        category_str = response.content.strip()
        category = int(category_str)

        # Валидация категории
        if category in VALID_CATEGORIES.keys():
            category_name = VALID_CATEGORIES[category]

            message = f"✓ Определена категория: {category_name}"
            return {
                "messages": [AIMessage(content=message)],
                "category": category,
                "category_name": category_name,
            }
        else:
            error_msg = f"✗ Получена невалидная категория: {category}"
            return {
                "messages": [AIMessage(content=error_msg)],
            }

    except Exception as e:
        logger.exception(f"Ошибка классификации: {e}")
        error_msg = "✗ Не удалось определить категорию проблемы"
        return {
            "messages": [AIMessage(content=error_msg)],
        }
