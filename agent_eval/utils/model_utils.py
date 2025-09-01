"""
Model utilities for consistent model handling across AgentEval.
"""

from typing import Any, Optional
from agent_eval.models.wrapper_factory import ModelWrapperFactory
from agent_eval.models.base_wrapper import BaseLLMWrapper


def wrap_model_safely(model: Any, model_name: Optional[str] = None, provider: Optional[str] = None) -> Optional[BaseLLMWrapper]:
    """
    Safely wrap a model with proper error handling and fallbacks.
    
    Args:
        model: The model to wrap (can be function, OpenAI client, etc.)
        model_name: Optional model name
        provider: Optional provider name
        
    Returns:
        Wrapped model or None if model is None
    """
    if model is None:
        return None
        
    if isinstance(model, BaseLLMWrapper):
        return model
    
    # Try to wrap the model using the factory
    try:
        return ModelWrapperFactory.wrap(model, model_name=model_name, provider=provider)
    except ValueError:
        # For functions or other callable types, create a simple wrapper
        if callable(model):
            class SimpleFunctionWrapper(BaseLLMWrapper):
                def __init__(self, func):
                    self.func = func
                    
                def generate(self, prompt: str, **kwargs) -> str:
                    return str(self.func(prompt))
            
            return SimpleFunctionWrapper(model)
        else:
            # Last resort - keep as-is and let downstream handle it
            return model
