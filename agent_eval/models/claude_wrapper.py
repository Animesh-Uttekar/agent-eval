from .base_wrapper import BaseLLMWrapper
from agent_eval.utils.logging_utils import loggable
from typing import Any


class ClaudeWrapper(BaseLLMWrapper):
    def __init__(self, client: Any, model_name: str = "claude-3-opus"):
        self.client = client
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            return (
                response.content
                if hasattr(response, "content")
                else response.choices[0].message.content
            )
        except Exception as e:
            raise RuntimeError(f"Claude generation failed: {str(e)}")
