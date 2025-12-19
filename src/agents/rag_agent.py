import logging
from typing import Any, Optional

from langchain.messages import AIMessage

from ..llm_factory import LLMFactory
from ..models.graph_state import GraphState

logger = logging.getLogger(__name__)


def rag_node(
    state: GraphState,
    llm_factory: LLMFactory,
    vector_store: Optional[Any] = None,
) -> dict:
    """
    Выполняет поиск в базе знаний и формирует ответ.

    На данный момент использует заглушку с mock-данными.
    В будущем будет интегрирована векторная БД.

    Args:
        state: текущее состояние с данными проблемы
        llm_factory: фабрика для создания LLM
        vector_store: векторное хранилище (пока не используется)

    Returns:
        dict с обновленным state: messages, rag_result
    """
    ttn_number = state.ttn_number
    problem_description = state.problem_description
    category_name = state.category_name

    logger.info(f"RAG поиск: ТТН={ttn_number}, категория={category_name}")

    # TODO: Интеграция с векторной БД
    # Сейчас используем заглушку
    if vector_store is None:
        logger.warning("Векторное хранилище не инициализировано, используется заглушка")
        mock_documents = _get_mock_documents(category_name)
    else:
        # В будущем: реальный поиск
        # results = vector_store.similarity_search(
        #     query=f"{category_name}: {problem_description}",
        #     k=3
        # )
        mock_documents = _get_mock_documents(category_name)

    # Создаем LLM для генерации ответа на основе найденных документов
    llm = llm_factory.create_llm()

    prompt = f"""На основе найденной информации из базы знаний ответь на проблему клиента.
        Проблема клиента:
        - Номер ТТН: {ttn_number}
        - Категория: {category_name}
        - Описание: {problem_description}
        
        Релевантная информация из базы знаний:
        {mock_documents}
        
        Сформируй подробный ответ с конкретными рекомендациями.
    """

    response = llm.invoke(prompt)
    rag_result = response.content

    logger.info("RAG поиск завершен, ответ сформирован")

    return {
        "messages": [AIMessage(content=rag_result)],
        "rag_result": rag_result,
    }


def _get_mock_documents(category_name: Optional[str]) -> str:
    """
    Возвращает mock-документы для тестирования.

    Args:
        category_name: название категории

    Returns:
        Форматированная строка с mock-документами
    """
    # Mock-данные для разных категорий
    mock_data = {}

    return mock_data.get(
        category_name or "",
        "**Общая информация:**\n\nПо данной категории в базе знаний пока нет "
        "специфической информации. Рекомендуется связаться с логистом для уточнения.",
    )
