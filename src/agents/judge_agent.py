"""Judge agent for evaluating dialog quality."""

import json
import logging
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import JUDGE_AGENT_PROMPT
from ..services.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class JudgeAgent:
    """
    Agent for evaluating DialogAgent performance.

    Evaluates based on:
    - Completeness of collected data
    - Relevance of questions asked
    - Efficiency of dialog (no unnecessary questions)
    """

    def __init__(self, llm_factory: LLMFactory):
        """
        Initialize judge agent.

        Args:
            llm_factory: Factory for creating LLM instances
        """
        self.llm_factory = llm_factory
        logger.info("JudgeAgent initialized")

    def evaluate_dialog(self, chat_history: list[dict[str, str]]) -> dict:
        """
        Evaluate dialog quality.

        Args:
            chat_history: List of messages with role and content

        Returns:
            Evaluation dict with score, completeness, issues, and summary
        """
        if not chat_history:
            logger.warning("No chat history to evaluate")
            return {
                "score": 0,
                "completeness": False,
                "issues": ["Нет истории диалога"],
                "summary": "Диалог пустой",
            }

        dialog_text = self._format_dialog(chat_history)
        evaluation_json = self._get_evaluation(dialog_text)

        return evaluation_json

    def _format_dialog(self, chat_history: list[dict[str, str]]) -> str:
        """
        Format chat history as text.

        Args:
            chat_history: List of messages

        Returns:
            Formatted dialog text
        """
        return "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])

    def _get_evaluation(self, dialog_text: str) -> dict:
        """
        Get LLM evaluation of dialog.

        Args:
            dialog_text: Formatted dialog text

        Returns:
            Evaluation dictionary
        """
        llm = self.llm_factory.create_llm()

        eval_prompt = f"{dialog_text}\n\nОцени диалог по критериям."
        messages = [
            SystemMessage(content=JUDGE_AGENT_PROMPT),
            HumanMessage(content=eval_prompt),
        ]

        try:
            logger.info("Requesting dialog evaluation from LLM")
            response = llm.invoke(messages)

            # Clean response (remove markdown code blocks if present)
            content = response.content.replace("```json", "").replace("```", "").strip()

            eval_json = json.loads(content)
            logger.info(f"Evaluation complete: score={eval_json.get('score')}")

            return eval_json

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            return {
                "score": 0,
                "completeness": False,
                "issues": ["Ошибка парсинга ответа агента"],
                "summary": f"Не удалось разобрать ответ: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {
                "score": 0,
                "completeness": False,
                "issues": [f"Ошибка оценки: {str(e)}"],
                "summary": "Произошла ошибка при оценке диалога",
            }

    def format_evaluation_markdown(self, evaluation: dict) -> str:
        """
        Format evaluation as markdown.

        Args:
            evaluation: Evaluation dictionary

        Returns:
            Markdown-formatted evaluation
        """
        issues = ", ".join(evaluation.get("issues", [])) or "Нет"
        completeness_mark = "✓" if evaluation.get("completeness") else "✗"

        return f"""
## Оценка
**Баллы:** {evaluation.get('score', 0)}/10

**Полнота:** {completeness_mark}

**Резюме:** {evaluation.get('summary', 'Нет резюме')}

**Проблемы:** {issues}
"""
