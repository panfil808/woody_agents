"""Data completeness checker tool."""

import logging
from langchain.tools import tool

logger = logging.getLogger(__name__)


@tool
def check_data_completeness(data: str) -> str:
    """
    Проверка полноты данных о проблеме.
    Ожидаемый формат: order_num|category|description|action

    Args:
        data: Данные в формате order_num|category|description|action

    Returns:
        Сообщение о результате проверки
    """
    if not data:
        return "✗ Данные отсутствуют"

    parts = data.split("|")

    if len(parts) != 4:
        return f"✗ Неверный формат. Ожидается 4 поля, получено {len(parts)}"

    order_num, category, description, action = parts

    missing = []
    if not order_num.strip():
        missing.append("номер заказа")
    if not category.strip():
        missing.append("категория")
    if not description.strip():
        missing.append("описание")
    if not action.strip():
        missing.append("действие")

    if missing:
        logger.warning(f"Incomplete data: missing {missing}")
        return f"✗ Не хватает: {', '.join(missing)}"

    logger.info("Data completeness check passed")
    return "✓ Все данные заполнены"
