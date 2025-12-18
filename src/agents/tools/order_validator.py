import logging
import re

from langchain.tools import tool

from ...config import config

logger = logging.getLogger(__name__)


@tool
def validate_ttn_number(ttn_number: str) -> str:
    """
    Проверка номера ТТН: должен быть 10 (если с ведущими нулями) или 8 цифр длиной,
    а также начинаться с 7 или 8

    Args:
        ttn_number: Номер ТТН для проверки

    Returns:
        Сообщение о результате валидации
    """
    if not ttn_number:
        return "✗ Номер заказа не указан"

    cleaned = ttn_number.strip()
    pattern = config.TTN_NUMBER_PATTERN
    is_valid = bool(re.match(pattern, cleaned))

    logger.info(f"TTN number validation: {cleaned} -> {is_valid}")

    if is_valid:
        return f"✓ Номер {cleaned} валиден"
    return f"✗ Номер {cleaned} невалиден"
