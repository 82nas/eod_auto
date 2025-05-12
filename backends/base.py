# /base.py
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class AIBaseBackend(ABC):
    """Abstract Base Class for AI backend implementations."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize backend with specific configuration."""
        self.config = config
        # Common initialization can go here if needed

    @abstractmethod
    def make_summary(self, task: str, ctx: str, arts: List[str]) -> str:
        """Generate the handover summary markdown."""
        pass

    @abstractmethod
    def verify_summary(self, md: str) -> Tuple[bool, str]:
        """Verify the generated markdown summary against rules."""
        pass

    @abstractmethod
    def load_report(self, md: str) -> str:
        """Generate a report based on understanding the loaded markdown."""
        pass

    @staticmethod
    def get_config_description() -> str:
        """Return a description of necessary config parameters for this backend."""
        return "특정 설정 없음." # Default message

    @staticmethod
    @abstractmethod # Each backend MUST provide its name
    def get_name() -> str:
        """Return the identifiable name of the backend."""
        pass