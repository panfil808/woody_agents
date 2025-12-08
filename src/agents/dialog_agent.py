"""Dialog agent for collecting customer problem information."""

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from .prompts import DIALOG_AGENT_SYSTEM_PROMPT
from ..services.llm_factory import LLMFactory
from ..tools.order_validator import validate_ttn_number
from ..tools.data_checker import check_data_completeness
from ..tools.problem_saver import save_problem_data

logger = logging.getLogger(__name__)


class DialogAgent:
    """
    Agent for conducting dialog with sellers to collect problem information.

    Follows a structured workflow:
    1. Collect order number and validate
    2. Collect problem details
    3. Collect required actions from logistician
    4. Signal completion for classification
    """

    def __init__(self, llm_factory: LLMFactory):
        """
        Initialize dialog agent.

        Args:
            llm_factory: Factory for creating LLM instances
        """
        self.llm_factory = llm_factory
        self._agent_graph = None
        self._checkpointer = InMemorySaver()

        logger.info("DialogAgent initialized")

    def get_agent_graph(self) -> Any:
        """
        Get or create agent graph.

        Returns:
            Agent graph instance
        """
        if self._agent_graph is None:
            self._agent_graph = self._create_agent_graph()
        return self._agent_graph

    def _create_agent_graph(self) -> Any:
        """
        Create LangChain agent graph with tools.

        Returns:
            Agent graph instance
        """
        llm = self.llm_factory.create_llm()

        tools = [
            validate_ttn_number,
            # check_data_completeness,
            # save_problem_data,
        ]

        logger.info("Creating agent graph with tools")

        return create_agent(
            model=llm,
            tools=tools,
            system_prompt=DIALOG_AGENT_SYSTEM_PROMPT,
            checkpointer=self._checkpointer,
        )

    def invoke(self, message: str, thread_id: str) -> str:
        """
        Process a message and return agent response.

        Args:
            message: User message
            thread_id: Thread ID for conversation tracking

        Returns:
            Agent response text
        """
        agent_graph = self.get_agent_graph()

        try:
            logger.info(f"Processing message for thread {thread_id}")

            response = agent_graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config={"configurable": {"thread_id": thread_id}},
            )

            messages = response.get("messages", [])
            if not messages:
                logger.warning("Agent returned no messages")
                return "(пустой ответ)"

            assistant_reply = messages[-1]
            content = assistant_reply.content

            logger.info(f"Agent response: {content[:100]}...")
            return content

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return f"Ошибка агента: {str(e)}"

    def reset(self) -> None:
        """Reset agent graph and checkpointer."""
        self._agent_graph = None
        self._checkpointer = InMemorySaver()
        logger.info("DialogAgent reset")
