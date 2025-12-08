"""
SmartWoody - Multi-agent customer support system for construction store sellers.

This is the main entry point for the application.
"""

import logging

from src.ui.gradio_interface import GradioInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    logger.info("Starting SmartWoody application")

    try:
        interface = GradioInterface()
        interface.launch()
    except Exception as e:
        logger.exception("Failed to start application")
        raise


if __name__ == "__main__":
    main()
