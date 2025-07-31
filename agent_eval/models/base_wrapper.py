from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLLMWrapper(ABC):
    """
    Abstract base wrapper for any LLM provider.
    All wrappers should implement this interface.
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from the model based on the given prompt.

        Args:
            prompt (str): The input prompt to the LLM.
            **kwargs (Any): Additional model-specific arguments.

        Returns:
            str: The generated output string.
        """
        pass
