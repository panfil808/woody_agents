"""Problem data saver tool."""

import json
import logging
from datetime import datetime
from langchain.tools import tool

from ..config import config
from ..models.problem_data import ProblemData

logger = logging.getLogger(__name__)


@tool
def save_problem_data(data: str) -> str:
    """
    Сохранение данных проблемы в JSON файл.
    Формат входных данных: order_num|category|description|action

    Args:
        data: Данные в формате order_num|category|description|action

    Returns:
        Сообщение о результате сохранения
    """
    try:
        # Парсим данные
        problem = ProblemData.from_pipe_delimited(data)

        if not problem.is_complete():
            return "✗ Данные неполные, не могу сохранить"

        # Создаем директорию если не существует
        output_dir = config.APPEALS_DIR
        output_dir.mkdir(exist_ok=True)

        # Генерируем имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"problem_{problem.order_number}_{timestamp}.json"
        filepath = output_dir / filename

        # Сохраняем в JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(problem.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Problem data saved to {filepath}")
        return f"✓ Данные сохранены в {filename}"

    except (ValueError, IOError) as e:
        logger.error(f"Failed to save problem data: {e}")
        return f"✗ Ошибка сохранения: {str(e)}"
