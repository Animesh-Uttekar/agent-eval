from agent_eval.models.openai_wrapper import OpenAIWrapper
from agent_eval.models.hf_wrapper import HuggingFaceWrapper
from agent_eval.models.mistral_wrapper import MistralWrapper
from agent_eval.models.claude_wrapper import ClaudeWrapper
from agent_eval.models.ollama_wrapper import OllamaWrapper
from agent_eval.models.cohere_wrapper import CohereWrapper
from agent_eval.models.base_wrapper import BaseLLMWrapper


class ModelWrapperFactory:
    """
    Factory class to wrap different models based on the provider or model type.
    """

    @staticmethod
    def wrap(model, model_name: str = None, provider: str = None) -> BaseLLMWrapper:
        """
        Wrap the model according to the provider or type.

        Args:
            model: The model to be wrapped.
            provider: The model provider, such as 'openai', 'huggingface', etc.
            model_name: Optional model name (e.g., "gpt-4", "command-r")

        Returns:
            BaseLLMWrapper: The wrapped model.
        """
        if isinstance(model, BaseLLMWrapper):
            return model

        if provider == "openai":
            return OpenAIWrapper(model, model_name=model_name)
        elif provider in ("hf", "huggingface"):
            return HuggingFaceWrapper(model, model_name=model_name)
        elif provider == "mistral":
            return MistralWrapper(model, model_name=model_name)
        elif provider == "claude":
            return ClaudeWrapper(model, model_name=model_name)
        elif provider == "ollama":
            return OllamaWrapper(model, model_name=model_name)
        elif provider == "cohere":
            return CohereWrapper(model, model_name=model_name)

        if hasattr(model, "chat") and hasattr(model.chat, "completions"):
            return OpenAIWrapper(model, model_name=model_name)
        elif hasattr(model, "__call__") and hasattr(model, "model"):
            return HuggingFaceWrapper(model, model_name=model_name)
        elif hasattr(model, "generate_sync"):
            return OllamaWrapper(model, model_name=model_name)
        elif hasattr(model, "messages") and hasattr(model, "completion"):
            return ClaudeWrapper(model, model_name=model_name)
        elif hasattr(model, "generate") and hasattr(model, "tokenize"):
            return CohereWrapper(model, model_name=model_name)

        raise ValueError(
            f"Unsupported model type: {type(model)}. Please provide the correct provider."
        )
