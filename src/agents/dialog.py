import logging

from langchain.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

from ..llm_factory import LLMFactory
from ..models.appeal_schema import AppealSchema
from ..models.graph_state import UnifiedGraphState
from .prompts import DIALOG_AGENT_SYSTEM_PROMPT
from .tools import validate_ttn_number

logger = logging.getLogger(__name__)


class DialogAgent:
    def __init__(self, llm_factory: LLMFactory):
        self._llm = llm_factory.create_llm()
        self._tools = [validate_ttn_number]
        self.tool_node = ToolNode(self._tools)

    def agent_node(self, state: UnifiedGraphState) -> dict:
        """Основной узел агента для обработки сообщений"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", DIALOG_AGENT_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        model = self._llm.bind_tools(self._tools)
        chain = prompt | model
        result = chain.invoke({"messages": state.messages})
        return {"messages": [result]}

    def structured_node(self, state: UnifiedGraphState) -> dict:
        """
        Извлекает структурированные данные из диалога.

        Записывает в state:
        - ttn_number: валидированный номер ТТН
        - problem_description: описание проблемы
        - dialog_complete: флаг завершения сбора информации
        """
        context = "\n".join([f"{msg.type}: {msg.content}" for msg in state.messages])
        extract_prompt = f"""Из следующего диалога извлеки ProblemSchema (ttn_number и problem_description).
            Диалог:
            {context}

            Убедись, что ttn_number валиден (проверен ранее), problem_description - полное описание проблемы.
            НЕ добавляй лишнего.
        """

        structured_llm = self._llm.with_structured_output(AppealSchema)
        schema = structured_llm.invoke(extract_prompt)

        content = ""
        complete = False
        if schema.ttn_number and schema.problem_description:
            content = f"✅ Информация собрана. Готово к классификации."
            complete = True
            logger.info(
                f"Structured data extracted: TTN={schema.ttn_number} PROBLEM: {schema.problem_description}"
            )

        return {
            "messages": [AIMessage(content=content)],
            "ttn_number": schema.ttn_number,
            "problem_description": schema.problem_description,
            "dialog_complete": complete,
        }

    @staticmethod
    def route_after_agent(state: UnifiedGraphState) -> str:
        tool_decision = tools_condition(state)
        if tool_decision == "tools":
            return "tools"

        if len(state.messages) == 0 or not isinstance(state.messages[-1], AIMessage):
            return END

        last_msg = state.messages[-1]
        if "готово к классификации" in last_msg.content.lower():
            return "structured"

        return END
