"""
Utility module for CLI formatting and shared helper functions.

This module provides reusable constants and functions to enhance
terminal output readability, such as ANSI color codes for logging.
"""

from typing import ClassVar
from pydantic import BaseModel


class TerminalColors(BaseModel):
    """
    ANSI escape sequences for terminal text formatting.

    Use these constants to wrap text strings in standard print statements
    or logging calls. Always append the `ENDC` constant at the end of the
    string to prevent color leaking into subsequent console outputs.

    Attributes:
        HEADER (ClassVar[str]): Purple color, ideal for titles or major
            section headers.
        OKBLUE (ClassVar[str]): Standard blue color for general informational
            messages.
        OKCYAN (ClassVar[str]): Cyan color for secondary information or
            variable highlights.
        OKGREEN (ClassVar[str]): Green color reserved for success states
            and validations.
        WARNING (ClassVar[str]): Yellow color for non-critical warnings or
            alerts.
        FAIL (ClassVar[str]): Red color for exceptions, errors, or critical
            failures.
        ENDC (ClassVar[str]): Reset code to return to the default terminal
            style.
        BOLD (ClassVar[str]): Formatting code to render text in bold weight.
        UNDERLINE (ClassVar[str]): Formatting code to underline text.
    """

    HEADER: ClassVar[str] = '\033[95m'
    OKBLUE: ClassVar[str] = '\033[94m'
    OKCYAN: ClassVar[str] = '\033[96m'
    OKGREEN: ClassVar[str] = '\033[92m'
    WARNING: ClassVar[str] = '\033[93m'
    FAIL: ClassVar[str] = '\033[91m'
    ENDC: ClassVar[str] = '\033[0m'
    BOLD: ClassVar[str] = '\033[1m'
    UNDERLINE: ClassVar[str] = '\033[4m'

    @staticmethod
    def info(message: str) -> None:
        """
        Print an informational message in bold cyan.
        """
        print(f"{TerminalColors.BOLD}{TerminalColors.OKCYAN}"
              f"[INFO] {message}{TerminalColors.ENDC}"
              )

    @staticmethod
    def success(message: str) -> None:
        """
        Print a success message in bold green.
        """
        print(f"{TerminalColors.BOLD}{TerminalColors.OKGREEN}"
              f" [SUCCESS] {message}{TerminalColors.ENDC}"
              )

    @staticmethod
    def warning(message: str) -> None:
        """
        Print a warning message in bold yellow.
        """
        print(f"{TerminalColors.BOLD}{TerminalColors.WARNING}"
              f" [WARNING] {message}{TerminalColors.ENDC}"
              )

    @staticmethod
    def error(message: str) -> None:
        """Print an error message in bold red."""
        print(f"{TerminalColors.BOLD}{TerminalColors.FAIL}"
              f" [ERROR] {message}{TerminalColors.ENDC}"
              )
