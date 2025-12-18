import logging
from typing import Any, Optional

from langchain.messages import AIMessage

from ..llm_factory import LLMFactory
from ..models.graph_state import UnifiedGraphState

logger = logging.getLogger(__name__)


def erp_node(
    state: UnifiedGraphState,
    llm_factory: LLMFactory,
    erp_client: Optional[Any] = None,
) -> dict:
    """
    Выполняет запрос к ERP-системе и формирует ответ.

    На данный момент использует заглушку с mock-данными.
    В будущем будет интегрирован реальный ERP API.

    Args:
        state: текущее состояние с данными проблемы
        llm_factory: фабрика для создания LLM
        erp_client: клиент для ERP API (пока не используется)

    Returns:
        dict с обновленным state: messages, erp_result
    """
    ttn_number = state.ttn_number
    problem_description = state.problem_description
    category_name = state.category_name

    logger.info(f"ERP запрос: ТТН={ttn_number}, категория={category_name}")

    # TODO: Интеграция с реальным ERP API
    # Сейчас используем заглушку
    if erp_client is None:
        logger.warning("ERP клиент не инициализирован, используется заглушка")
        erp_data = _get_mock_erp_data(ttn_number, category_name)
    else:
        # В будущем: реальный запрос к ERP
        # erp_data = erp_client.get_order_info(ttn_number)
        erp_data = _get_mock_erp_data(ttn_number, category_name)

    # Создаем LLM для генерации ответа на основе данных из ERP
    llm = llm_factory.create_llm()

    prompt = f"""На основе данных из ERP системы предложи решение проблемы клиента.    
        Проблема клиента:
        - Номер ТТН: {ttn_number}
        - Категория: {category_name}
        - Описание: {problem_description}
        
        Данные из ERP системы:
        {erp_data}
        
        Сформируй конкретный план действий для решения проблемы.
    """

    response = llm.invoke(prompt)
    erp_result = response.content

    logger.info("ERP запрос завершен, ответ сформирован")

    return {
        "messages": [AIMessage(content=erp_result)],
        "erp_result": erp_result,
    }


def _get_mock_erp_data(ttn_number: Optional[str], category_name: Optional[str]) -> str:
    """
    Возвращает mock-данные из ERP для тестирования.

    Args:
        ttn_number: номер ТТН
        category_name: название категории

    Returns:
        Форматированная строка с mock-данными из ERP
    """
    # Mock-данные о заказе
    return ""
