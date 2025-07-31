from agent_eval.models.base_wrapper import BaseLLMWrapper
from agent_eval.utils.logging_utils import loggable
from typing import Optional


class OllamaWrapper(BaseLLMWrapper):
    """
    Wrapper for Ollama models to standardize usage across the evaluation framework.
    """

    def __init__(self, client, model_name: Optional[str] = None):
        self.client = client
        self.model_name = model_name or "llama3"

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": kwargs.get("temperature", 0.2),
                "num_predict": kwargs.get("max_tokens", 1000),
            },
        )
        return response["message"]["content"]
