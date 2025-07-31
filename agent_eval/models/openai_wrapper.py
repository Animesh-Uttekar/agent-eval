from typing import Optional
from agent_eval.models.base_wrapper import BaseLLMWrapper
from agent_eval.utils.logging_utils import loggable


class OpenAIWrapper(BaseLLMWrapper):
    def __init__(self, client, model_name: Optional[str] = None):
        self.client = client
        self.model_name = model_name or getattr(client, "model_name", "gpt-3.5-turbo")

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1000),
        )
        return response.choices[0].message.content
