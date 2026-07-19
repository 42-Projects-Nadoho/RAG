"""
Utility module for CLI formatting and shared helper functions.

This module provides reusable constants and functions to enhance
terminal output readability, such as ANSI color codes for logging.
"""

from pydantic import BaseModel


class TerminalColors(BaseModel):
    """
    ANSI escape sequences for terminal text formatting.

    Use these constants to wrap text strings in standard print statements
    or logging calls. Always append the `ENDC` constant at the end of the
    string to prevent color leaking into subsequent console outputs.

    Attributes:
        HEADER (str): Purple color, ideal for titles or major section headers.
        OKBLUE (str): Standard blue color for general informational messages.
        OKCYAN (str): Cyan color for secondary information or variable
        highlights.
        OKGREEN (str): Green color reserved for success states and validations.
        WARNING (str): Yellow color for non-critical warnings or alerts.
        FAIL (str): Red color for exceptions, errors, or critical failures.
        ENDC (str): Reset code to return to the default terminal style.
        BOLD (str): Formatting code to render text in bold weight.
        UNDERLINE (str): Formatting code to underline text.
    """

    HEADER: str = '\033[95m'
    OKBLUE: str = '\033[94m'
    OKCYAN: str = '\033[96m'
    OKGREEN: str = '\033[92m'
    WARNING: str = '\033[93m'
    FAIL: str = '\033[91m'
    ENDC: str = '\033[0m'
    BOLD: str = '\033[1m'
    UNDERLINE: str = '\033[4m'

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
