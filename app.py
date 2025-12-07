import os

import gradio as gr
import logging
import json
import re
import uuid
from datetime import datetime
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.environ["OPENAI_API_KEY"]


# === Инструменты ===
@tool
def validate_order_number(order_num: str) -> str:
    """
    Проверка номера заказа: должен быть 10 (если с ведущими нулями) или 8 цифр длиной,
    а также начинаться с 7 или 8
    """
    if re.match(r'^0{2}[7|8]\d{7}|[7|8]\d{7}$', order_num.strip()):
        return f"✓ Номер {order_num} валиден"
    return f"✗ Номер {order_num} невалиден"

@tool
def check_data_completeness(data: str) -> str:
    """Проверка полноты собранных данных.
    Формат: order_num|category|description|action"""
    parts = data.split("|")
    if len(parts) < 4:
        return "incomplete"
    order, cat, desc, action = parts[0], parts[1], parts[2], parts[3]
    if order and cat and desc and action:
        return "complete"
    return "incomplete"

@tool
def save_problem_data(data: str) -> str:
    """Сохранение данных проблемы в JSON.
    Формат: order_num|category|description|action"""
    try:
        parts = data.split("|")
        problem_data = {
            "timestamp": datetime.now().isoformat(),
            "order_number": parts[0],
            "category": parts[1],
            "description": parts[2],
            "required_action": parts[3]
        }
        filename = f"problem_{parts[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(problem_data, f, ensure_ascii=False, indent=2)
        return f"✓ Данные сохранены в {filename}"
    except Exception as e:
        return f"✗ Ошибка сохранения: {str(e)}"


# === PROMPTS ===
DIALOG_AGENT_SYSTEM_PROMPT = """Ты DialogAgent — помощник для продавцов строительных магазинов.

АЛГОРИТМ:
1. Узнай категорию проблемы (доставка/комплектация/брак/возврат)
2. Запроси номер заказа и проверь через validate_order_number
3. Выясни детали проблемы и требуемые действия
4. Проверь полноту через check_data_completeness (формат: номер|категория|описание|действие)
5. Если complete — сохрани через save_problem_data

ПРАВИЛА:
- Задавай по 1 вопросу за раз
- Не предлагай решения сам
- Используй инструменты для валидации

У тебя есть доступ к следующим инструментам:"""

JUDGE_AGENT_PROMPT = """Ты JudgeAgent — оцениваешь качество работы DialogAgent.

Критерии:
1. Полнота (номер заказа, категория, описание, требуемые действия)
2. Релевантность вопросов (соответствуют ли вопросы проблеме)
3. Эффективность (не было ли лишних вопросов)

ОТВЕТ СТРОГО JSON (без markdown):
{{"score": 8, "completeness": true, "issues": ["..."], "summary": "..."}}

Где:
- score: число от 1 до 10
- completeness: true или false
- issues: массив строк (может быть пустым)
- summary: краткая оценка текстом
"""

def create_llm(model_name: str, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        temperature=0.3,
        max_tokens=1000,
        timeout=30,
        api_key=api_key,
        base_url="https://gpt.sdvor.com/api/v1",
    )


def create_dialog_agent(api_key: str):
    """Создание DialogAgent с инструментами"""
    logger.info("Создание DialogAgent")
    llm = create_llm("cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit", api_key)

    return create_agent(
        model=llm,
        tools=[validate_order_number, check_data_completeness, save_problem_data],
        system_prompt=DIALOG_AGENT_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def process_message(
        message: str,
        history: list[dict],
        state: dict[str, Any]
) -> tuple[list[dict], dict[str, Any]]:
    logger.info("process_message: start, active=%s, has_api_key=%s", state.get("active"), bool(state.get("api_key")))
    logger.info("process_message: incoming message='%s'", message)
    api_key = state.get("api_key")
    if not api_key:
        return history + [{"role": "assistant", "content": "❌ Инициализируйте систему"}], state

    if not state.get("active"):
        state["active"] = True
        state["chat_history"] = []
        state["agent_graph"] = create_dialog_agent(api_key)
        state["thread_id"] = str(uuid.uuid4())
        logger.info("process_message: new session started, thread_id=%s", state["thread_id"])

    agent_graph = state["agent_graph"]
    thread_id = state.setdefault("thread_id", str(uuid.uuid4()))
    logger.info("process_message: using thread_id=%s", thread_id)

    try:
        response = agent_graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        logger.info("process_message: agent_graph.invoke OK, response_keys=%s", list(response.keys()))
        messages = response.get("messages", [])
        assistant_reply = messages[-1] if messages else AIMessage(content="(пустой ответ)")
        content = assistant_reply.content
        logger.info("process_message: assistant_reply='%s'", content)
    except Exception as e:
        logger.exception("process_message: Ошибка агента")
        content = f"Ошибка агента: {str(e)}"

    state["chat_history"].append({"role": "user", "content": message})
    state["chat_history"].append({"role": "assistant", "content": content})

    return state["chat_history"], state


def initialize_system(api_key: str, state: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    key = api_key or API_KEY
    if not key:
        return [{"role": "assistant", "content": "❌ Введите API ключ"}], {}

    try:
        logger.info("initialize_system: проверка ключа и модели")
        llm = create_llm("cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit", key)
        llm.invoke("test")
        new_state = {"api_key": key, "active": False, "chat_history": [], "thread_id": None}
        logger.info("initialize_system: успешно, состояние инициализировано")
        return [{"role": "assistant", "content": "✅ Система готова"}], new_state
    except Exception as e:
        logger.exception("initialize_system: ошибка инициализации")
        return [{"role": "assistant", "content": f"❌ {str(e)}"}], state


def new_session(state: dict[str, Any]) -> tuple[list[dict], dict[str, Any]]:
    if not state.get("api_key"):
        return [{"role": "assistant", "content": "❌ Инициализируйте систему"}], state
    new_state = {"api_key": state["api_key"], "active": False, "chat_history": [], "thread_id": None}
    logger.info("new_session: создано новое состояние, старый thread_id=%s", state.get("thread_id"))
    return [{"role": "assistant", "content": "🔄 Новая сессия"}], new_state


def end_session(state: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if not state.get("active"):
        return "❌ Нет сессии", "", state

    chat_history = state["chat_history"]
    dialog_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])

    # Summary
    summary = f"Диалог из {len(chat_history)} сообщений"

    # Judge
    llm = create_llm("cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit", state["api_key"])
    eval_prompt = f"{dialog_text}\n\nОцени диалог по критериям."
    messages = [SystemMessage(content=JUDGE_AGENT_PROMPT), HumanMessage(content=eval_prompt)]
    response = llm.invoke(messages)
    content = response.content.replace("```json", "").replace("```", "").strip()

    try:
        eval_json = json.loads(content)
        issues = ", ".join(eval_json.get("issues", [])) or "Нет"
        eval_md = f"""
            ## Оценка
            **Баллы:** {eval_json.get('score')}/10\n
            **Полнота:** {'✓' if eval_json.get('completeness') else '✗'}\n
            **Резюме:** {eval_json.get('summary')}\n
            **Проблемы:** {issues}
        """
    except:
        eval_md = f"Ошибка парсинга:\n{content[:200]}"

    state["active"] = False
    state["thread_id"] = None
    logger.info("end_session: завершение сессии, итоговый summary='%s'", summary)
    return summary, eval_md, state


# === GRADIO ===
with gr.Blocks(title="Мультиагентная система для поддержки продавцов") as demo:
    gr.Markdown("# 🏗️ Мультиагентная система (LangChain Agents + Tools)")

    with gr.Row():
        api_key_input = gr.Textbox(label="API ключ", type="password", scale=3)
        init_btn = gr.Button("Инициализировать", variant="primary", scale=1)

    chatbot = gr.Chatbot(label="Диалог", height=400)
    session_state = gr.State({})

    with gr.Row():
        msg = gr.Textbox(label="Сообщение", scale=4)
        submit_btn = gr.Button("Отправить", variant="primary", scale=1)

    with gr.Row():
        new_session_btn = gr.Button("🔄 Новая сессия")
        end_btn = gr.Button("✅ Завершить")

    with gr.Row():
        summary_out = gr.Textbox(label="Резюме", lines=4)
        eval_out = gr.Markdown(label="Оценка")

    init_btn.click(initialize_system, [api_key_input, session_state], [chatbot, session_state])
    submit_btn.click(process_message, [msg, chatbot, session_state], [chatbot, session_state]).then(lambda *args: "",
                                                                                                    None, msg)
    msg.submit(process_message, [msg, chatbot, session_state], [chatbot, session_state]).then(lambda *args: "", None,
                                                                                              msg)
    new_session_btn.click(new_session, session_state, [chatbot, session_state]).then(lambda *args: ("", ""), None,
                                                                                     [summary_out, eval_out])
    end_btn.click(end_session, session_state, [summary_out, eval_out, session_state])

if __name__ == "__main__":
    demo.launch(ssr_mode=False, debug=True, server_name="0.0.0.0")
