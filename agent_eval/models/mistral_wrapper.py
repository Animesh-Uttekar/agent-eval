from agent_eval.models.base_wrapper import BaseLLMWrapper
from agent_eval.utils.logging_utils import loggable
from typing import Union, Dict


class MistralWrapper(BaseLLMWrapper):
    """
    Wrapper for Mistral inference models. Accepts a callable that generates text.
    """

    def __init__(self, inference_fn: callable, model_name: str = "mistral-7b"):
        self.inference_fn = inference_fn
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        kwargs.setdefault("temperature", 0.2)
        kwargs.setdefault("max_new_tokens", 512)
        response = self.inference_fn(prompt, **kwargs)
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return response.get("text") or response.get("output") or str(response)
        else:
            return str(response)
