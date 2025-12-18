import logging

from langchain_openai import ChatOpenAI

from .config import config

logger = logging.getLogger(__name__)


class LLMFactory:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = None,
    ):
        """
        Initialize LLM factory.

        Args:
            api_key: API key for authentication (defaults to config)
            base_url: Base URL for API (defaults to config)
            model: Model name (defaults to config)
            temperature: Sampling temperature (defaults to config)
            max_tokens: Maximum tokens in response (defaults to config)
            timeout: Request timeout in seconds (defaults to config)
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = base_url or config.LLM_BASE_URL
        self.model = model or config.LLM_MODEL
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.timeout = timeout or config.LLM_TIMEOUT

    def create_llm(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> ChatOpenAI:
        """
        Create a new LLM instance.

        Args:
            model: Override model name
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            ChatOpenAI instance
        """
        final_model = model or self.model
        final_temperature = temperature if temperature is not None else self.temperature
        final_max_tokens = max_tokens or self.max_tokens

        logger.info(
            f"Creating LLM instance: model={final_model}, temp={final_temperature}"
        )

        return ChatOpenAI(
            model=final_model,
            temperature=final_temperature,
            max_tokens=final_max_tokens,
            timeout=self.timeout,
            api_key=self.api_key,
            base_url=self.base_url,
        )
