import logging
from typing import Any, Optional

from ..models.session_state import SessionState

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages conversation sessions and their state"""

    def __init__(self):
        """Initialize session manager."""
        self._sessions: dict[str, SessionState] = {}

    def get_session(self, session_id: str = "default") -> Optional[SessionState]:
        """
        Get existing session.

        Args:
            session_id: Unique session identifier

        Returns:
            SessionState instance or None if not found
        """
        return self._sessions.get(session_id)

    def start_session(self, session_id: str, agent_graph: Any) -> bool:
        """
        Start an existing session.

        Args:
            session_id: Unique session identifier
            agent_graph: Agent graph instance

        Returns:
            True if session started, False if session not found
        """
        state = self.get_session(session_id)
        if state is None:
            logger.warning(f"Cannot start session {session_id}: not found")
            return False

        state.start_session(agent_graph)
        logger.info(f"Started session {session_id} with thread_id={state.thread_id}")
        return True

    def end_session(self, session_id: str = "default") -> bool:
        """
        End a session.

        Args:
            session_id: Unique session identifier

        Returns:
            True if session ended, False if session not found
        """
        state = self.get_session(session_id)
        if state is None:
            logger.warning(f"Cannot end session {session_id}: not found")
            return False

        state.end_session()
        logger.info(f"Ended session {session_id}")
        return True
