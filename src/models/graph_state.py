import operator
from typing import Annotated

from langchain.messages import AnyMessage
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    messages: Annotated[
        list[AnyMessage],
        operator.add,
        Field(
            default_factory=list, description="История переписки, общая для всех агентов"
        ),
    ]

    # Выходные данные DialogAgent
    ttn_number: str | None = Field(
        None, description="Валидированный номер ТТН от DialogAgent"
    )
    problem_description: str | None = Field(
        None, description="Описание проблемы, собранное DialogAgent"
    )

    # Выходные данные ClassifierAgent
    category: str = Field(None, description="Категория проблемы")
    category_name: str | None = Field(None, description="Читаемое имя категории")

    # Результаты специализированных агентов
    rag_result: str | None = Field(
        None, description="Результат от RAG-агента (поиск в базе знаний)"
    )
    erp_result: str | None = Field(
        None, description="Результат от ERP-агента (интеграция с системой)"
    )
    final_appeal: str | None = Field(
        None, description="Финальное сформированное обращение для логистов"
    )

    # Управление workflow
    dialog_complete: bool = Field(
        False, description="Флаг завершения сбора информации диалоговым агентом"
    )
    routing_target: str | None = Field(
        None, description="Целевой агент для маршрутизации: 'rag', 'erp' или 'finalizer'"
    )
