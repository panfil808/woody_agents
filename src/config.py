"""Configuration management for SmartWoody."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration class."""

    # LLM Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://gpt.sdvor.com/api/v1")
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL", "cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
    )
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))

    # Application Configuration
    APPEALS_DIR: Path = Path(os.getenv("APPEALS_DIR", "appeals"))
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "7861"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "True").lower() == "true"

    # Order number validation regex
    ORDER_NUMBER_PATTERN: str = r"^0{2}[7|8]\d{7}|[7|8]\d{7}$"

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")

        # Create appeals directory if it doesn't exist
        cls.APPEALS_DIR.mkdir(exist_ok=True)


# Global config instance
config = Config()
