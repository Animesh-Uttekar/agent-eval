from .base_wrapper import BaseLLMWrapper
from agent_eval.utils.logging_utils import loggable


class HuggingFaceWrapper(BaseLLMWrapper):
    def __init__(self, pipeline, model_name: str = None):
        self.pipeline = pipeline
        self._model_name = model_name or getattr(
            pipeline.model, "name_or_path", "hf-default-model"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        outputs = self.pipeline(prompt, **kwargs)
        if isinstance(outputs, list):
            out = outputs[0]
            return out.get("generated_text") or out.get("text") or str(out)
        elif isinstance(outputs, str):
            return outputs
        return str(outputs)

    @property
    def model_name(self) -> str:
        return self._model_name
