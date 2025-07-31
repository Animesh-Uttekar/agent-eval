from agent_eval.models.base_wrapper import BaseLLMWrapper
from agent_eval.utils.logging_utils import loggable


class CohereWrapper(BaseLLMWrapper):
    """
    Wrapper for Cohere's command-x models.
    """

    def __init__(self, cohere_client, model_name: str = "command-r+"):
        self.client = cohere_client
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.generate(prompt=prompt, model=self.model_name, **kwargs)
        if hasattr(response, "generations"):
            return response.generations[0].text
        elif isinstance(response, str):
            return response
        else:
            return str(response)
