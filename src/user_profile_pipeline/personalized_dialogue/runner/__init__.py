"""Modular implementation behind the personalized-dialogue runner script."""

from .cli import build_parser
from .orchestration import run

__all__ = ["build_parser", "run"]
