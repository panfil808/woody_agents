import logging

from langchain.messages import AIMessage

from ..llm_factory import LLMFactory
from ..models.graph_state import UnifiedGraphState

logger = logging.getLogger(__name__)


def finalizer_node(
    state: UnifiedGraphState,
    llm_factory: LLMFactory,
) -> dict:
    ttn_number = state.ttn_number
    problem_description = state.problem_description
    category_name = state.category_name
    llm = llm_factory.create_llm()

    prompt = f"""
        На основе собранной информации сформируй краткое обращение для логистов.
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

    return {
        "messages": [AIMessage(content=final_appeal)],
        "final_appeal": final_appeal,
    }
