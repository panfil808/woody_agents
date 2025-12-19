import logging
from typing import Any

import gradio as gr
from langchain.messages import AIMessage, HumanMessage

from ..agents.graph_state import build_unified_graph
from ..config import config
from ..llm_factory import LLMFactory
from ..models.session_state import SessionState

logger = logging.getLogger(__name__)


class GradioInterface:
    def create_interface(self) -> gr.Blocks:
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
        api_key = state.get("api_key") or config.OPENAI_API_KEY

        if not api_key:
            return (
                history + [AIMessage(content="❌ API ключ не настроен в .env файле")],
                state,
            )

        state["api_key"] = api_key

        if not state.get("active"):
            state["active"] = True
            state["chat_history"] = []

            llm_factory = LLMFactory(api_key=api_key)
            state["llm_factory"] = llm_factory

            graph = build_unified_graph(
                llm_factory=llm_factory,
                vector_store=None,  # Заглушка, позже добавим ChromaDB/FAISS
                erp_client=None,  # Заглушка, позже добавим ERP API
            )
            state["graph"] = graph

            state["session_state"] = SessionState(api_key=api_key)
            state["session_state"].start_session(graph)

        session_state = state["session_state"]
        graph = state["graph"]

        try:
            response = graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config={
                    "configurable": {"thread_id": session_state.get_or_create_thread_id()}
                },
            )

            all_messages = response.get("messages", [])
            category = response.get("category")

            session_state.add_message("user", message)

            new_ai_messages = [msg for msg in all_messages if isinstance(msg, AIMessage)]
            if new_ai_messages:
                session_state.add_message("assistant", new_ai_messages[-1])

            if category:
                session_state.set_category(category)

            state["chat_history"] = session_state.chat_history

            return state["chat_history"], state

        except Exception as e:
            logger.exception("Error processing message with unified graph")
            error_msg = f"Ошибка системы: {str(e)}"

            session_state.add_message("user", message)
            session_state.add_message("assistant", error_msg)

            return session_state.chat_history, state

    @staticmethod
    def new_session(state: dict[str, Any]) -> tuple[list[AIMessage], dict[str, Any]]:
        api_key = state.get("api_key") or config.OPENAI_API_KEY
        if not api_key:
            return [AIMessage(content="❌ API ключ не настроен")], state

        new_state = {
            "api_key": api_key,
            "active": False,
            "chat_history": [],
        }

        return [AIMessage(content="❌ API ключ не настроен")], new_state

    def launch(
        self,
        server_name: str = None,
        server_port: int = None,
        debug: bool = None,
    ) -> None:
        demo = self.create_interface()

        demo.launch(
            server_name=server_name or config.SERVER_HOST,
            server_port=server_port or config.SERVER_PORT,
            debug=debug or config.DEBUG_MODE,
            ssr_mode=False,
        )
