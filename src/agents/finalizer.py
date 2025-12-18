import logging

from langchain.messages import AIMessage

from ..llm_factory import LLMFactory
from ..models.graph_state import UnifiedGraphState

logger = logging.getLogger(__name__)


def finalizer_node(
    state: UnifiedGraphState,
    llm_factory: LLMFactory,
) -> dict:
    """
    Формирует итоговое обращение для логистов.

    Этот агент используется для категорий, которые не требуют
    обращения к RAG или ERP-системе. Просто форматирует собранную
    информацию в удобный вид для передачи логистам.

    Args:
        state: Текущее состояние с собранными данными
        llm_factory: Фабрика для создания LLM

    Returns:
        dict с обновленным state: messages, final_appeal
    """
    ttn_number = state.ttn_number
    problem_description = state.problem_description
    category_name = state.category_name

    logger.info(f"Финализация обращения: ТТН={ttn_number}, категория={category_name}")

    # Создаем LLM для форматирования обращения
    llm = llm_factory.create_llm()

    prompt = f"""На основе собранной информации сформируй краткое обращение для логистов.
        Данные обращения:
        - Номер ТТН: {ttn_number}
        - Категория: {category_name}
        - Описание проблемы: {problem_description}
        
        Сформируй структурированное обращение в формате:
        
        **Обращение по транспортировке {ttn_number}**
        Категория: {category_name}
        
        Описание:
        {problem_description}
    """

    response = llm.invoke(prompt)
    final_appeal = response.content

    logger.info("Обращение сформировано успешно")

    return {
        "messages": [AIMessage(content=final_appeal)],
        "final_appeal": final_appeal,
    }
