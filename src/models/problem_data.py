"""Problem data model for customer issues."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ProblemData:
    """Data class for customer problem information."""

    order_number: str
    category: str
    description: str
    required_action: str
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "order_number": self.order_number,
            "category": self.category,
            "description": self.description,
            "required_action": self.required_action,
        }

    @classmethod
    def from_pipe_delimited(cls, data: str) -> "ProblemData":
        """
        Create ProblemData from pipe-delimited string.

        Args:
            data: String in format "order_num|category|description|action"

        Returns:
            ProblemData instance

        Raises:
            ValueError: If data format is invalid
        """
        parts = data.split("|")
        if len(parts) < 4:
            raise ValueError(
                f"Invalid data format. Expected 4 parts, got {len(parts)}"
            )

        return cls(
            order_number=parts[0].strip(),
            category=parts[1].strip(),
            description=parts[2].strip(),
            required_action=parts[3].strip(),
        )

    def to_pipe_delimited(self) -> str:
        """Convert to pipe-delimited string format."""
        return f"{self.order_number}|{self.category}|{self.description}|{self.required_action}"

    def is_complete(self) -> bool:
        """Check if all required fields are filled."""
        return bool(
            self.order_number
            and self.category
            and self.description
            and self.required_action
        )
