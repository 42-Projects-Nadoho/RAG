"""
Main entry point for the RAG against the machine project CLI.
"""

import fire
from src.pipeline import RagPipeline
from src.models.utils import TerminalColors


def main() -> None:
    """
    Entry point for the fire CLI application.
    """
    fire.Fire(RagPipeline)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        TerminalColors.error("Keyboard interruption was detected.")
