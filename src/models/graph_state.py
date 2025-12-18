import operator
from typing import Annotated, Optional

from langchain.messages import AnyMessage
from pydantic import BaseModel, Field


class UnifiedGraphState(BaseModel):
    """
    Единое состояние для мультиагентной LangGraph системы.

    Это состояние разделяется между всеми агентами в workflow:
    - DialogAgent собирает ttn_number и problem_description
    - ClassifierAgent определяет категорию
    - RAG/ERP/Finalizer обрабатывают в зависимости от категории
    """

    # История сообщений (автоматически конкатенируется через operator.add)
    messages: Annotated[list[AnyMessage], operator.add] = Field(
        default_factory=list, description="История переписки, общая для всех агентов"
    )

    # Выходные данные DialogAgent
    ttn_number: Optional[str] = Field(
        None, description="Валидированный номер ТТН от DialogAgent"
    )
    problem_description: Optional[str] = Field(
        None, description="Описание проблемы, собранное DialogAgent"
    )

    # Выходные данные ClassifierAgent
    category: Optional[int] = Field(
        None,
        ge=1,
        le=4,
        description="Категория проблемы (1-4), будет расширена в будущем",
    )
    category_name: Optional[str] = Field(None, description="Читаемое имя категории")

    # Результаты специализированных агентов
    rag_result: Optional[str] = Field(
        None, description="Результат от RAG-агента (поиск в базе знаний)"
    )
    erp_result: Optional[str] = Field(
        None, description="Результат от ERP-агента (интеграция с системой)"
    )
    final_appeal: Optional[str] = Field(
        None, description="Финальное сформированное обращение для логистов"
    )

    # Управление workflow
    dialog_complete: bool = Field(
        False, description="Флаг завершения сбора информации DialogAgent"
    )
    routing_target: Optional[str] = Field(
        None, description="Целевой агент для маршрутизации: 'rag', 'erp' или 'finalizer'"
    )

    class Config:
        arbitrary_types_allowed = True  # Требуется для AnyMessage
