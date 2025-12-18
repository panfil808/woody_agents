import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionState:
    api_key: str
    active: bool = False
    chat_history: list[dict[str, str]] = field(default_factory=list)
    agent_graph: Optional[Any] = None
    thread_id: Optional[str] = None
    category: Optional[str] = None

    def start_session(self, agent_graph: Any) -> None:
        self.active = True
        self.chat_history = []
        self.agent_graph = agent_graph
        self.thread_id = str(uuid.uuid4())

    def add_message(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})

    def get_or_create_thread_id(self) -> str:
        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())
        return self.thread_id

    def set_category(self, category: str) -> None:
        self.category = category
