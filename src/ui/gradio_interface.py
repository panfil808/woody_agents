import logging
from typing import Any

import gradio as gr
from langchain.messages import HumanMessage, AIMessage

from ..agents.unified_graph import build_unified_graph
from ..config import config
from ..llm_factory import LLMFactory
from ..models.session_state import SessionState

logger = logging.getLogger(__name__)


class GradioInterface:
    def __init__(self):
        logger.info("GradioInterface initialized")

    def create_interface(self) -> gr.Blocks:
        """
        Create and configure Gradio interface.

        Returns:
            Gradio Blocks instance
        """
        with gr.Blocks(title="Мультиагентная система поддержки продавцов") as demo:
            gr.Markdown("# 🏗️ Мультиагентная система поддержки продавцов")

            chatbot = gr.Chatbot(label="Диалог", height=550)
            session_state = gr.State({})

            with gr.Row():
                msg = gr.Textbox(
                    label="Сообщение", scale=4, placeholder="Опишите проблему..."
                )
                submit_btn = gr.Button("Отправить", variant="primary", scale=1)

            new_session_btn = gr.Button("🔄 Новая сессия")

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
            )

        return demo

    @staticmethod
    def process_message(
        message: str, history: list[dict], state: dict[str, Any]
    ) -> tuple[list[dict], dict[str, Any]]:
        """
        Process user message and return agent response

        Args:
            message: User message
            history: Chat history
            state: Current session state

        Returns:
            Tuple of (updated_history, updated_state)
        """
        logger.info(f"Processing message, active={state.get('active')}")

        api_key = state.get("api_key")
        if not api_key:
            api_key = config.OPENAI_API_KEY
            if not api_key:
                return (
                    history
                    + [
                        {
                            "role": "assistant",
                            "content": "❌ API ключ не настроен в .env файле",
                        }
                    ],
                    state,
                )

            state["api_key"] = api_key
            logger.info("Auto-initialized with API key from config")

        if not state.get("active"):
            state["active"] = True
            state["chat_history"] = []

            # Создаем единый граф для всех агентов
            llm_factory = LLMFactory(api_key=api_key)
            state["llm_factory"] = llm_factory

            # Строим unified graph с заглушками для RAG и ERP
            # TODO: Интегрировать реальные vector_store и erp_client
            unified_graph = build_unified_graph(
                llm_factory=llm_factory,
                vector_store=None,  # Заглушка, позже добавим ChromaDB/FAISS
                erp_client=None,  # Заглушка, позже добавим ERP API
            )
            state["unified_graph"] = unified_graph

            state["session_state"] = SessionState(api_key=api_key)
            state["session_state"].start_session(unified_graph)

            logger.info("New session created with unified graph")

        session_state = state["session_state"]
        unified_graph = state["unified_graph"]

        try:
            # Единый вызов графа - все агенты выполняются автоматически
            logger.info(
                f"Invoking unified graph for thread {session_state.get_or_create_thread_id()}"
            )

            response = unified_graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config={
                    "configurable": {"thread_id": session_state.get_or_create_thread_id()}
                },
            )

            # Извлекаем результаты из state
            all_messages = response.get("messages", [])
            category = response.get("category")
            rag_result = response.get("rag_result")
            erp_result = response.get("erp_result")
            final_appeal = response.get("final_appeal")

            # Обновляем историю сообщений
            session_state.add_message("user", message)

            # Извлекаем ТОЛЬКО новые сообщения агента (AIMessage) из этого вызова
            # response["messages"] содержит ВСЕ сообщения (старые + новые)
            # Нам нужны только новые AIMessage, сгенерированные в этом вызове

            new_ai_messages = []
            for msg in all_messages:
                if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                    new_ai_messages.append(msg.content)

            # Добавляем только ПОСЛЕДНЕЕ сообщение агента (самое новое)
            # Это избегает дублирования старых сообщений
            if new_ai_messages:
                session_state.add_message("assistant", new_ai_messages[-1])

            # Сохраняем категорию, если была определена
            if category:
                session_state.set_category(category)
                logger.info(f"Problem classified as category: {category}")

            # Логируем результаты специализированных агентов
            if rag_result:
                logger.info("RAG agent provided result")
            if erp_result:
                logger.info("ERP agent provided result")
            if final_appeal:
                logger.info("Finalizer created appeal")

            state["chat_history"] = session_state.chat_history
            logger.info(f"Всего сообщений в истории: {len(state['chat_history'])}")

            return state["chat_history"], state

        except Exception as e:
            logger.exception("Error processing message with unified graph")
            error_msg = f"Ошибка системы: {str(e)}"

            session_state.add_message("user", message)
            session_state.add_message("assistant", error_msg)

            return session_state.chat_history, state

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
                return [
                    {"role": "assistant", "content": "❌ API ключ не настроен"}
                ], state

        new_state = {
            "api_key": api_key,
            "active": False,
            "chat_history": [],
        }

        logger.info("New session created")
        return [
            {"role": "assistant", "content": "🔄 Новая сессия. Начните диалог."}
        ], new_state

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
