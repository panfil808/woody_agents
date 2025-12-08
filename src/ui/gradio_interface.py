"""Gradio interface for SmartWoody."""

import logging
from typing import Any

import gradio as gr

from ..agents.dialog_agent import DialogAgent
from ..agents.judge_agent import JudgeAgent
from ..agents.classifier_agent import ClassifierAgent
from ..services.llm_factory import LLMFactory
from ..models.session_state import SessionState
from ..config import config

logger = logging.getLogger(__name__)


class GradioInterface:
    """Gradio web interface for the SmartWoody system."""

    def __init__(self):
        """Initialize Gradio interface."""
        self.current_state: dict[str, Any] = {}
        logger.info("GradioInterface initialized")

    def create_interface(self) -> gr.Blocks:
        """
        Create and configure Gradio interface.

        Returns:
            Gradio Blocks instance
        """
        with gr.Blocks(title="Мультиагентная система для поддержки продавцов") as demo:
            gr.Markdown("# 🏗️ Мультиагентная система для поддержки продавцов")

            chatbot = gr.Chatbot(label="Диалог", height=550)
            session_state = gr.State({})

            with gr.Row():
                msg = gr.Textbox(label="Сообщение", scale=4, placeholder="Опишите проблему...")
                submit_btn = gr.Button("Отправить", variant="primary", scale=1)

            with gr.Row():
                new_session_btn = gr.Button("🔄 Новая сессия")
                end_btn = gr.Button("✅ Завершить")

            with gr.Row():
                summary_out = gr.Textbox(label="Резюме", lines=4)
                eval_out = gr.Markdown(label="Оценка")

            # Event handlers
            submit_btn.click(
                self.process_message,
                [msg, chatbot, session_state],
                [chatbot, session_state],
            ).then(lambda *args: "", None, msg)

            msg.submit(
                self.process_message,
                [msg, chatbot, session_state],
                [chatbot, session_state],
            ).then(lambda *args: "", None, msg)

            new_session_btn.click(
                self.new_session, session_state, [chatbot, session_state]
            ).then(lambda *args: ("", ""), None, [summary_out, eval_out])

            end_btn.click(
                self.end_session, session_state, [summary_out, eval_out, session_state]
            )

        return demo

    @staticmethod
    def process_message(
            message: str, history: list[dict], state: dict[str, Any]
    ) -> tuple[list[dict], dict[str, Any]]:
        """
        Process user message and return agent response.

        Args:
            message: User message
            history: Chat history
            state: Current session state

        Returns:
            Tuple of (updated_history, updated_state)
        """
        logger.info(f"Processing message, active={state.get('active')}")

        # Auto-initialize on first message if needed
        api_key = state.get("api_key")
        if not api_key:
            # Use API key from environment
            api_key = config.OPENAI_API_KEY
            if not api_key:
                return history + [
                    {"role": "assistant", "content": "❌ API ключ не настроен в .env файле"}
                ], state

            state["api_key"] = api_key
            logger.info("Auto-initialized with API key from config")

        # Start new session if not active
        if not state.get("active"):
            state["active"] = True
            state["chat_history"] = []
            state["collection_data"] = {}  # Store collected data

            # Create agents
            llm_factory = LLMFactory(api_key=api_key)
            state["llm_factory"] = llm_factory
            state["classifier_agent"] = ClassifierAgent(llm_factory)

            # Create DialogAgent immediately
            dialog_agent = DialogAgent(llm_factory)
            state["dialog_agent"] = dialog_agent

            state["session_state"] = SessionState(api_key=api_key)
            state["session_state"].start_session(dialog_agent.get_agent_graph())

            logger.info("New session created, DialogAgent ready to collect information")

        session_state = state["session_state"]
        dialog_agent = state["dialog_agent"]

        try:
            # DialogAgent collects information
            response = dialog_agent.invoke(message, session_state.get_thread_id())

            session_state.add_message("user", message)
            session_state.add_message("assistant", response)

            # Check if DialogAgent signals data collection is complete
            if "готово к классификации" in response.lower() or "информация собрана" in response.lower():
                logger.info("DialogAgent signaled data collection complete, invoking ClassifierAgent")

                # Extract data from chat history for classification
                order_number, problem_desc, required_actions = GradioInterface._extract_data_from_history(
                    session_state.chat_history
                )

                if order_number and problem_desc and required_actions:
                    classifier_agent = state["classifier_agent"]
                    category, classification_msg = classifier_agent.classify_from_data(
                        order_number, problem_desc, required_actions
                    )

                    if category:
                        session_state.set_category(category)
                        session_state.add_message("assistant", classification_msg)
                        logger.info(f"Problem classified as: {category}")
                    else:
                        session_state.add_message("assistant", "✗ Не удалось определить категорию")
                        logger.warning("Classification failed")
                else:
                    logger.warning("Cannot classify - missing data")
                    session_state.add_message("assistant", "⚠️ Недостаточно данных для классификации")

            state["chat_history"] = session_state.chat_history
            return state["chat_history"], state

        except Exception as e:
            logger.exception("Error processing message")
            error_msg = f"Ошибка агента: {str(e)}"

            session_state.add_message("user", message)
            session_state.add_message("assistant", error_msg)

            return session_state.chat_history, state

    @staticmethod
    def _extract_data_from_history(chat_history: list[dict]) -> tuple[str, str, str]:
        """
        Extract order number, problem description, and required actions from chat history.

        Args:
            chat_history: List of chat messages

        Returns:
            Tuple of (order_number, problem_description, required_actions)
        """
        order_number = ""
        problem_desc = ""
        required_actions = ""

        # Simple heuristic: extract from conversation
        full_text = " ".join([msg["content"] for msg in chat_history if msg["role"] == "user"])

        # Try to find order number pattern
        import re
        order_match = re.search(r'\b(?:0{2})?[7|8]\d{7}\b', full_text)
        if order_match:
            order_number = order_match.group()

        # Problem description and actions - combine relevant user messages
        user_messages = [msg["content"] for msg in chat_history if msg["role"] == "user"]
        if len(user_messages) >= 2:
            problem_desc = " ".join(user_messages[:-1])  # All but last
            required_actions = user_messages[-1] if user_messages else ""
        elif user_messages:
            problem_desc = user_messages[0]
            required_actions = "Уточнить у клиента"

        logger.info(
            f"Extracted data: order={order_number}, "
            f"desc_len={len(problem_desc)}, actions_len={len(required_actions)}"
        )

        return order_number, problem_desc, required_actions

    @staticmethod
    def new_session(state: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
        """
        Start a new session.

        Args:
            state: Current session state

        Returns:
            Tuple of (empty_history, updated_state)
        """
        api_key = state.get("api_key")
        if not api_key:
            # Try to get from config
            api_key = config.OPENAI_API_KEY
            if not api_key:
                return [{"role": "assistant", "content": "❌ API ключ не настроен"}], state

        new_state = {
            "api_key": api_key,
            "active": False,
            "chat_history": [],
        }

        logger.info("New session created")
        return [{"role": "assistant", "content": "🔄 Новая сессия. Начните диалог."}], new_state

    @staticmethod
    def end_session(
            state: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any]]:
        """
        End current session and evaluate dialog.

        Args:
            state: Current session state

        Returns:
            Tuple of (summary, evaluation_markdown, updated_state)
        """
        if not state.get("active"):
            return "❌ Нет сессии", "", state

        chat_history = state.get("chat_history", [])
        summary = f"Диалог из {len(chat_history)} сообщений"

        # Evaluate dialog
        try:
            llm_factory = LLMFactory(api_key=state["api_key"])
            judge_agent = JudgeAgent(llm_factory)

            evaluation = judge_agent.evaluate_dialog(chat_history)
            eval_md = judge_agent.format_evaluation_markdown(evaluation)

        except Exception as e:
            logger.exception("Evaluation failed")
            eval_md = f"Ошибка оценки: {str(e)}"

        # End session
        state["active"] = False
        if "session_state" in state:
            state["session_state"].end_session()

        logger.info("Session ended")

        return summary, eval_md, state

    def launch(
        self,
        server_name: str = None,
        server_port: int = None,
        debug: bool = None,
    ) -> None:
        """
        Launch Gradio interface.

        Args:
            server_name: Server host (defaults to config)
            server_port: Server port (defaults to config)
            debug: Debug mode (defaults to config)
        """
        demo = self.create_interface()

        demo.launch(
            server_name=server_name or config.SERVER_HOST,
            server_port=server_port or config.SERVER_PORT,
            debug=debug if debug is not None else config.DEBUG_MODE,
            ssr_mode=False,
        )
