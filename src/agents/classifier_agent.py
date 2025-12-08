"""Classifier agent for categorizing customer issues based on collected information."""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..services.llm_factory import LLMFactory
from .prompts import CLASSIFIER_AGENT_PROMPT

logger = logging.getLogger(__name__)


class CategoriesEnum:
    SEND_EARLIER = "Отправить доставку пораньше"
    UNASSIGN_DRIVER = "Снять водителя с доставки"
    CANCEL_DELIVERY = "Отменить доставку"
    DEMAND_DRIVER_INFO = "Запросить данные водителя для пропуска"


class ClassifierAgent:
    """
    Agent for classifying problem category based on collected information.
    """

    VALID_CATEGORIES = {
        1: CategoriesEnum.SEND_EARLIER,
        2: CategoriesEnum.UNASSIGN_DRIVER,
        3: CategoriesEnum.CANCEL_DELIVERY,
        4: CategoriesEnum.DEMAND_DRIVER_INFO,
    }

    def __init__(self, llm_factory: LLMFactory):
        """
        Initialize classifier agent.

        Args:
            llm_factory: Factory for creating LLM instances
        """
        self.llm_factory = llm_factory
        logger.info("ClassifierAgent initialized")

    def classify(
        self, order_number: str, problem_description: str, required_actions: str
    ) -> Optional[int]:
        """
        Classify problem into one of predefined categories based on collected data.

        Args:
            order_number: Customer's order number
            problem_description: Detailed description of the problem
            required_actions: Actions required from logistician

        Returns:
            Category string or None if classification failed
        """
        if not all([order_number, problem_description, required_actions]):
            logger.warning("Incomplete information for classification")
            return None

        llm = self.llm_factory.create_llm()

        # Format collected information for classification
        collected_info = f"""
            Номер ТТН: {order_number}
            
            Описание проблемы:
            {problem_description}
            
            Требуемые действия от логиста:
            {required_actions}
        """

        messages = [
            SystemMessage(content=CLASSIFIER_AGENT_PROMPT),
            HumanMessage(content=collected_info),
        ]

        try:
            logger.info("Classifying problem description")
            response = llm.invoke(messages)
            category = response.content.strip()

            # Validate category
            if int(category) in self.VALID_CATEGORIES.keys():
                logger.info(f"Problem classified as: {category}")
                return int(category)

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            return None

    def classify_from_data(
        self, order_number: str, problem_description: str, required_actions: str
    ) -> tuple[Optional[int], str]:
        """
        Classify problem and return both category and user-friendly message.

        Args:
            order_number: Customer's order number
            problem_description: Detailed description of the problem
            required_actions: Actions required from logistician

        Returns:
            Tuple of (category, message)
        """
        category = self.classify(order_number, problem_description, required_actions)

        if category:
            readable_name = self.VALID_CATEGORIES.get(category, category)
            message = f"✓ Определена категория: {readable_name}"
            return category, message
        else:
            message = "✗ Не удалось определить категорию проблемы. Пожалуйста, опишите проблему подробнее."
            return None, message
