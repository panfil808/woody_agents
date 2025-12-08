"""Session manager for handling conversation sessions."""

import logging
from typing import Any, Optional

from ..models.session_state import SessionState

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages conversation sessions and their state."""

    def __init__(self):
        """Initialize session manager."""
        self._sessions: dict[str, SessionState] = {}

    def create_session(self, api_key: str, session_id: str = "default") -> SessionState:
        """
        Create a new session.

        Args:
            api_key: API key for LLM calls
            session_id: Unique session identifier

        Returns:
            SessionState instance
        """
        state = SessionState(api_key=api_key)
        self._sessions[session_id] = state

        logger.info(f"Created new session: {session_id}")
        return state

    def get_session(self, session_id: str = "default") -> Optional[SessionState]:
        """
        Get existing session.

        Args:
            session_id: Unique session identifier

        Returns:
            SessionState instance or None if not found
        """
        return self._sessions.get(session_id)

    def start_session(
        self,
        session_id: str,
        agent_graph: Any
    ) -> bool:
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

    def reset_session(self, session_id: str = "default") -> bool:
        """
        Reset a session (keep API key, clear history).

        Args:
            session_id: Unique session identifier

        Returns:
            True if session reset, False if session not found
        """
        state = self.get_session(session_id)
        if state is None:
            logger.warning(f"Cannot reset session {session_id}: not found")
            return False

        api_key = state.api_key
        self._sessions[session_id] = SessionState(api_key=api_key)

        logger.info(f"Reset session {session_id}")
        return True

    def delete_session(self, session_id: str = "default") -> bool:
        """
        Delete a session completely.

        Args:
            session_id: Unique session identifier

        Returns:
            True if session deleted, False if session not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True

        logger.warning(f"Cannot delete session {session_id}: not found")
        return False
