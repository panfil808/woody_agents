import logging

from src.ui.gradio_interface import GradioInterface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting application")

    try:
        interface = GradioInterface()
        interface.launch()
    except Exception as e:
        logger.exception("Failed to start application")
        raise e


if __name__ == "__main__":
    main()
