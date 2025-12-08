import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionState:
    """Session state for managing agent conversations"""

    api_key: str
    active: bool = False
    chat_history: list[dict[str, str]] = field(default_factory=list)
    agent_graph: Optional[Any] = None
    thread_id: Optional[str] = None
    category: Optional[str] = None

    def start_session(self, agent_graph: Any) -> None:
        """
        Start a new session.

        Args:
            agent_graph: The agent graph instance to use
        """
        self.active = True
        self.chat_history = []
        self.agent_graph = agent_graph
        self.thread_id = str(uuid.uuid4())

    def end_session(self) -> None:
        self.active = False
        self.thread_id = None
        self.category = None

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to chat history.

        Args:
            role: Message role (user/assistant)
            content: Message content
        """
        self.chat_history.append({"role": role, "content": content})

    def get_thread_id(self) -> str:
        """
        Get thread ID, creating one if it doesn't exist.

        Returns:
            Thread ID string
        """
        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())
        return self.thread_id

    def set_category(self, category: str) -> None:
        """
        Set problem category.

        Args:
            category: Problem category
        """
        self.category = category
