"""Mock model wrapper for testing purposes."""

from agent_eval.models.base_wrapper import BaseLLMWrapper


class MockWrapper(BaseLLMWrapper):
    """Wrapper for MockModel used in testing."""
    
    def __init__(self, model, model_name: str = "mock-model"):
        self.model = model
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using the mock model."""
        return self.model.generate(prompt)
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow callable interface."""
        return self.generate(prompt, **kwargs)