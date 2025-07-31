from agent_eval.models.base_wrapper import BaseLLMWrapper
from agent_eval.utils.logging_utils import loggable
from typing import Dict


class GeminiWrapper(BaseLLMWrapper):
    """
    Wrapper for Google's Gemini API (via Vertex AI or similar).
    """

    def __init__(self, gemini_client, model_name: str = "gemini-pro"):
        self.client = gemini_client
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.generate_text(prompt=prompt, **kwargs)
        if hasattr(response, "text"):
            return response.text
        elif isinstance(response, str):
            return response
        else:
            return str(response)
