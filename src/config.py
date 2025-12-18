import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://gpt.sdvor.com/api/v1")
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL", "cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
    )
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "30"))

    # Application
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "7860"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", True)

    # TTN number validation regex
    TTN_NUMBER_PATTERN: str = r"^0{2}[7|8]\d{7}|[7|8]\d{7}$"

    @classmethod
    def validate(cls) -> None:
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")


config = Config()
