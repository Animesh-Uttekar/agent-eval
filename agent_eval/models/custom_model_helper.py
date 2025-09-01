"""
Custom Model Integration Helper for AgentEval

Provides utilities and examples for integrating proprietary and custom models
with AgentEval's evaluation system without requiring any changes to the core codebase.
"""

from typing import Any, Dict, Optional, Callable, Union, List
from abc import ABC, abstractmethod
import asyncio
import requests
import json
from agent_eval.models.base_wrapper import BaseLLMWrapper


class ProprietaryModelWrapper(BaseLLMWrapper):
    """
    Generic wrapper for     # Example 2: Local proprietary model  
    def create_local_model():
        # Your model loading logic (example)
        # your_model = load_your_proprietary_model()
        # your_tokenizer = load_your_tokenizer()
        
        # Mock for example purposes
        your_model = None  # Replace with your actual model
        your_tokenizer = None  # Replace with your actual tokenizer
        
        # Example preprocessing function
        def custom_preprocessing(prompt):
            return f"[CUSTOM] {prompt}"
        
        # Example postprocessing function  
        def custom_postprocessing(response):
            return response.strip()
        
        return create_local_model_wrapper(
            model_instance=your_model,
            tokenizer=your_tokenizer,
            generation_method="inference",  # or whatever method your model uses
            preprocessing_fn=custom_preprocessing,
            postprocessing_fn=custom_postprocessing
        )odels accessed via API endpoints.
    
    Supports common patterns for proprietary model integration:
    - Custom authentication (API keys, OAuth, custom headers)
    - Custom request/response formats
    - Rate limiting and retry logic
    - Custom parameter mapping
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str = None,
        model_name: str = "proprietary-model",
        custom_headers: Dict[str, str] = None,
        request_format: str = "openai",  # "openai", "anthropic", "custom"
        auth_type: str = "bearer",  # "bearer", "api_key", "custom"
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize proprietary model wrapper.
        
        Args:
            endpoint: API endpoint URL
            api_key: Authentication key
            model_name: Name of the model for identification
            custom_headers: Additional headers to include in requests
            request_format: Request format template ("openai", "anthropic", "custom")
            auth_type: Authentication type ("bearer", "api_key", "custom")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_format = request_format
        self.auth_type = auth_type
        
        # Setup headers
        self.headers = {"Content-Type": "application/json"}
        if custom_headers:
            self.headers.update(custom_headers)
        
        # Setup authentication
        if api_key and auth_type == "bearer":
            self.headers["Authorization"] = f"Bearer {api_key}"
        elif api_key and auth_type == "api_key":
            self.headers["X-API-Key"] = api_key
        
        # Setup session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using proprietary model API."""
        
        # Build request payload based on format
        if self.request_format == "openai":
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
                "top_p": kwargs.get("top_p", 1.0)
            }
        elif self.request_format == "anthropic":
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
        else:  # custom format
            payload = self._build_custom_payload(prompt, **kwargs)
        
        # Make API call with retries
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Parse response based on format
                return self._parse_response(response.json())
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Proprietary model API call failed after {self.max_retries} attempts: {str(e)}")
                # Wait before retry (exponential backoff)
                import time
                time.sleep(2 ** attempt)
    
    def _build_custom_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build custom payload format. Override this method for custom APIs."""
        return {
            "input": prompt,
            "parameters": {
                "temperature": kwargs.get("temperature", 0.7),
                "max_length": kwargs.get("max_tokens", 1000),
                **kwargs  # Include any additional custom parameters
            }
        }
    
    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """Parse API response. Override this method for custom response formats."""
        
        # Try common response formats
        if "choices" in response_data and len(response_data["choices"]) > 0:
            # OpenAI format
            choice = response_data["choices"][0]
            if "message" in choice:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]
        
        if "content" in response_data:
            # Anthropic format
            if isinstance(response_data["content"], list):
                return response_data["content"][0].get("text", "")
            return response_data["content"]
        
        if "generated_text" in response_data:
            # HuggingFace format
            return response_data["generated_text"]
        
        if "output" in response_data:
            # Generic output format
            return response_data["output"]
        
        if "text" in response_data:
            # Simple text format
            return response_data["text"]
        
        # Fallback to string representation
        return str(response_data)


class LocalModelWrapper(BaseLLMWrapper):
    """
    Wrapper for local proprietary models (e.g., custom-trained models, optimized deployments).
    
    Supports:
    - Local model instances
    - Custom preprocessing/postprocessing pipelines
    - GPU memory management
    - Batch optimization
    """
    
    def __init__(
        self,
        model_instance: Any,
        tokenizer: Any = None,
        preprocessing_fn: Callable[[str], Any] = None,
        postprocessing_fn: Callable[[Any], str] = None,
        generation_method: str = "generate",  # Method name to call on model
        model_name: str = "local-proprietary-model"
    ):
        """
        Initialize local model wrapper.
        
        Args:
            model_instance: Your loaded model instance
            tokenizer: Your tokenizer instance (if needed)
            preprocessing_fn: Custom preprocessing function
            postprocessing_fn: Custom postprocessing function
            generation_method: Method name to call for generation
            model_name: Model identifier
        """
        self.model = model_instance
        self.tokenizer = tokenizer
        self.preprocessing_fn = preprocessing_fn or self._default_preprocess
        self.postprocessing_fn = postprocessing_fn or self._default_postprocess
        self.generation_method = generation_method
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using local proprietary model."""
        
        # Apply custom preprocessing
        processed_input = self.preprocessing_fn(prompt)
        
        # Get generation method from model
        generate_fn = getattr(self.model, self.generation_method)
        
        # Call model with processed input
        raw_output = generate_fn(processed_input, **kwargs)
        
        # Apply custom postprocessing
        final_output = self.postprocessing_fn(raw_output)
        
        return final_output
    
    def _default_preprocess(self, prompt: str) -> Any:
        """Default preprocessing - tokenize if tokenizer available."""
        if self.tokenizer:
            return self.tokenizer.encode(prompt, return_tensors="pt")
        return prompt
    
    def _default_postprocess(self, output: Any) -> str:
        """Default postprocessing - decode if tokenizer available."""
        if self.tokenizer and hasattr(output, 'sequences'):
            # Handle typical transformer outputs
            return self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        elif self.tokenizer and hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(output, skip_special_tokens=True)
        
        # Handle various output formats
        if isinstance(output, dict):
            if "generated_text" in output:
                return output["generated_text"]
            elif "text" in output:
                return output["text"]
        elif isinstance(output, list) and len(output) > 0:
            return str(output[0])
        
        return str(output)


class AsyncModelWrapper(BaseLLMWrapper):
    """
    Wrapper for asynchronous proprietary models.
    
    Converts async model calls to sync for AgentEval compatibility.
    """
    
    def __init__(
        self,
        async_model: Any,
        async_method: str = "generate_async",
        model_name: str = "async-proprietary-model"
    ):
        self.async_model = async_model
        self.async_method = async_method
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Synchronously generate response from async model."""
        
        # Get async generation method
        async_generate_fn = getattr(self.async_model, self.async_method)
        
        # Run async function synchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Call async function
        result = loop.run_until_complete(async_generate_fn(prompt, **kwargs))
        
        return str(result)


class BatchModelWrapper(BaseLLMWrapper):
    """
    Wrapper for batch-optimized proprietary models.
    
    Optimizes single requests for models designed for batch processing.
    """
    
    def __init__(
        self,
        batch_model: Any,
        batch_method: str = "batch_generate",
        model_name: str = "batch-proprietary-model",
        min_batch_size: int = 1,
        max_wait_time: float = 0.1
    ):
        self.batch_model = batch_model
        self.batch_method = batch_method
        self.model_name = model_name
        self.min_batch_size = min_batch_size
        self.max_wait_time = max_wait_time
        self._pending_requests = []
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using batch model."""
        
        # For single requests, create a batch of 1
        batch_generate_fn = getattr(self.batch_model, self.batch_method)
        
        # Call batch function with single prompt
        batch_results = batch_generate_fn([prompt], **kwargs)
        
        # Return first result
        if isinstance(batch_results, list) and len(batch_results) > 0:
            return str(batch_results[0])
        
        return str(batch_results)


class MultiStageModelWrapper(BaseLLMWrapper):
    """
    Wrapper for multi-stage proprietary model pipelines.
    
    Supports complex pipelines with multiple models or processing stages.
    """
    
    def __init__(
        self,
        pipeline_stages: List[Callable[[str], str]],
        model_name: str = "multi-stage-proprietary-model"
    ):
        """
        Initialize multi-stage wrapper.
        
        Args:
            pipeline_stages: List of functions/models in pipeline order
            model_name: Model identifier
        """
        self.pipeline_stages = pipeline_stages
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response through multi-stage pipeline."""
        
        current_output = prompt
        
        # Pass through each stage of the pipeline
        for i, stage in enumerate(self.pipeline_stages):
            try:
                if i == 0:
                    # First stage gets the original prompt
                    current_output = stage(current_output, **kwargs)
                else:
                    # Subsequent stages get output from previous stage
                    current_output = stage(current_output)
            except Exception as e:
                raise Exception(f"Pipeline stage {i} failed: {str(e)}")
        
        return str(current_output)


def create_custom_wrapper(
    model: Any,
    wrapper_type: str = "auto",
    **config
) -> BaseLLMWrapper:
    """
    Factory function to create appropriate wrapper for custom models.
    
    Args:
        model: Your proprietary model instance
        wrapper_type: Type of wrapper ("api", "local", "async", "batch", "multi_stage", "auto")
        **config: Configuration options specific to wrapper type
        
    Returns:
        Appropriate wrapper for the model
    """
    
    if wrapper_type == "api":
        return ProprietaryModelWrapper(
            endpoint=config.get("endpoint"),
            api_key=config.get("api_key"),
            model_name=config.get("model_name", "proprietary-api-model"),
            **config
        )
    
    elif wrapper_type == "local":
        return LocalModelWrapper(
            model_instance=model,
            **config
        )
    
    elif wrapper_type == "async":
        return AsyncModelWrapper(
            async_model=model,
            **config
        )
    
    elif wrapper_type == "batch":
        return BatchModelWrapper(
            batch_model=model,
            **config
        )
    
    elif wrapper_type == "multi_stage":
        return MultiStageModelWrapper(
            pipeline_stages=config.get("pipeline_stages", [model]),
            **config
        )
    
    elif wrapper_type == "auto":
        # Auto-detect appropriate wrapper
        if isinstance(model, str) and (model.startswith("http") or model.startswith("https")):
            # Treat as API endpoint
            return ProprietaryModelWrapper(
                endpoint=model,
                api_key=config.get("api_key"),
                **config
            )
        elif hasattr(model, "generate_async") or hasattr(model, "agenerate"):
            # Async model
            return AsyncModelWrapper(async_model=model, **config)
        elif hasattr(model, "batch_generate"):
            # Batch model
            return BatchModelWrapper(batch_model=model, **config)
        else:
            # Default to local wrapper
            return LocalModelWrapper(model_instance=model, **config)
    
    else:
        raise ValueError(f"Unknown wrapper_type: {wrapper_type}")


# Convenience functions for common proprietary model patterns

def create_api_model_wrapper(
    endpoint: str,
    api_key: str,
    model_name: str = None,
    **kwargs
) -> ProprietaryModelWrapper:
    """Create wrapper for API-based proprietary models."""
    return ProprietaryModelWrapper(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model_name or "proprietary-api-model",
        **kwargs
    )


def create_local_model_wrapper(
    model_instance: Any,
    tokenizer: Any = None,
    **kwargs
) -> LocalModelWrapper:
    """Create wrapper for local proprietary models."""
    return LocalModelWrapper(
        model_instance=model_instance,
        tokenizer=tokenizer,
        **kwargs
    )


def create_function_wrapper(
    generation_function: Callable[[str], str],
    model_name: str = "custom-function-model"
) -> BaseLLMWrapper:
    """
    Create wrapper for simple generation functions.
    
    Perfect for wrapping existing inference functions without changes.
    """
    
    class FunctionWrapper(BaseLLMWrapper):
        def __init__(self, func, name):
            self.func = func
            self.model_name = name
        
        def generate(self, prompt: str, **kwargs) -> str:
            try:
                # Try calling with kwargs
                return self.func(prompt, **kwargs)
            except TypeError:
                # Fallback to prompt only
                return self.func(prompt)
    
    return FunctionWrapper(generation_function, model_name)


# Example usage patterns
def example_usage():
    """
    Examples showing how to integrate various proprietary model types.
    """
    
    # Example 1: API-based proprietary model
    def create_api_model():
        return create_api_model_wrapper(
            endpoint="https://your-proprietary-model.com/api/v1/generate",
            api_key="your-secret-api-key",
            model_name="your-model-v2.1",
            request_format="openai",  # or "anthropic" or "custom"
            custom_headers={"X-Custom-Header": "value"}
        )
    
    # Example 2: Local proprietary model
    def create_local_model():
        # Your model loading logic (replace with your actual implementation)
        # your_model = load_your_proprietary_model()
        # your_tokenizer = load_your_tokenizer()
        
        # Mock for example purposes
        your_model = None  # Replace with your actual model
        your_tokenizer = None  # Replace with your actual tokenizer
        
        # Example preprocessing function
        def custom_preprocessing(prompt):
            return f"[CUSTOM] {prompt}"
        
        # Example postprocessing function
        def custom_postprocessing(response):
            return response.strip()
        
        return create_local_model_wrapper(
            model_instance=your_model,
            tokenizer=your_tokenizer,
            generation_method="inference",  # or whatever method your model uses
            preprocessing_fn=custom_preprocessing,
            postprocessing_fn=custom_postprocessing
        )
    
    # Example 3: Existing inference function
    def create_function_model():
        def your_existing_inference(prompt, temperature=0.7, max_tokens=1000):
            # Your existing inference logic here
            return "Your model's response"
        
        return create_function_wrapper(
            generation_function=your_existing_inference,
            model_name="your-existing-function"
        )
    
    # Example 4: Complex multi-stage pipeline
    def create_pipeline_model():
        def reasoning_stage(prompt):
            return f"Reasoned: {prompt}"
        
        def fact_check_stage(intermediate):
            return f"Fact-checked: {intermediate}"
        
        def safety_filter_stage(final):
            return f"Safe: {final}"
        
        return create_custom_wrapper(
            model=None,
            wrapper_type="multi_stage",
            pipeline_stages=[reasoning_stage, fact_check_stage, safety_filter_stage],
            model_name="multi-stage-pipeline"
        )


# Utility functions for proprietary model integration

def test_model_wrapper(wrapper: BaseLLMWrapper, test_prompt: str = "Hello, how are you?") -> bool:
    """
    Test if a custom model wrapper works correctly.
    
    Args:
        wrapper: Custom model wrapper to test
        test_prompt: Test prompt to send
        
    Returns:
        True if wrapper works correctly, False otherwise
    """
    try:
        response = wrapper.generate(test_prompt)
        return isinstance(response, str) and len(response) > 0
    except Exception as e:
        print(f"Wrapper test failed: {str(e)}")
        return False


def create_proprietary_model_config(
    model_type: str,
    **model_specific_config
) -> Dict[str, Any]:
    """
    Create configuration for proprietary model integration.
    
    Args:
        model_type: Type of proprietary model ("api", "local", "distributed", etc.)
        **model_specific_config: Model-specific configuration
        
    Returns:
        Configuration dictionary for the model
    """
    
    base_config = {
        "model_name": "proprietary-model",
        "timeout": 30,
        "max_retries": 3,
        "enable_caching": True
    }
    
    if model_type == "api":
        base_config.update({
            "request_format": "openai",
            "auth_type": "bearer",
            "custom_headers": {}
        })
    elif model_type == "local":
        base_config.update({
            "generation_method": "generate",
            "batch_size": 1,
            "device": "auto"
        })
    
    base_config.update(model_specific_config)
    return base_config


# Ready-to-use templates for common proprietary model providers

class CommonProprietaryModels:
    """Pre-configured wrappers for common proprietary model patterns."""
    
    @staticmethod
    def openai_compatible_api(endpoint: str, api_key: str, model_name: str = None):
        """OpenAI-compatible API wrapper."""
        return ProprietaryModelWrapper(
            endpoint=endpoint,
            api_key=api_key,
            model_name=model_name or "openai-compatible",
            request_format="openai",
            auth_type="bearer"
        )
    
    @staticmethod
    def anthropic_compatible_api(endpoint: str, api_key: str, model_name: str = None):
        """Anthropic-compatible API wrapper."""
        return ProprietaryModelWrapper(
            endpoint=endpoint,
            api_key=api_key,
            model_name=model_name or "anthropic-compatible", 
            request_format="anthropic",
            auth_type="api_key"
        )
    
    @staticmethod
    def azure_openai(endpoint: str, api_key: str, deployment_name: str, api_version: str = "2024-02-01"):
        """Azure OpenAI Service wrapper."""
        return ProprietaryModelWrapper(
            endpoint=f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}",
            api_key=api_key,
            model_name=f"azure-{deployment_name}",
            request_format="openai",
            auth_type="api_key",
            custom_headers={"api-key": api_key}
        )
    
    @staticmethod
    def custom_internal_api(endpoint: str, auth_token: str, model_name: str = None):
        """Internal/corporate API wrapper."""
        return ProprietaryModelWrapper(
            endpoint=endpoint,
            api_key=auth_token,
            model_name=model_name or "internal-api-model",
            request_format="custom",
            auth_type="bearer",
            custom_headers={"X-Internal-Auth": auth_token}
        )


# Integration examples with AgentEval
def integration_examples():
    """Show how to use custom wrappers with AgentEval."""
    
    from agent_eval import evaluator
    
    # Example 1: Proprietary API model
    api_model = CommonProprietaryModels.openai_compatible_api(
        endpoint="https://your-model-api.com/v1/chat/completions",
        api_key="your-api-key",
        model_name="your-proprietary-gpt-4"
    )
    
    eval_instance = evaluator(model=api_model, domain="healthcare")
    
    # Example 2: Local proprietary model
    def load_your_model():
        # Your model loading logic
        pass
    
    local_model = create_local_model_wrapper(
        model_instance=load_your_model(),
        model_name="your-local-model"
    )
    
    eval_instance = evaluator(model=local_model)
    
    # Example 3: Existing function
    def your_existing_inference(prompt):
        # Your existing logic
        return "Response"
    
    function_model = create_function_wrapper(your_existing_inference)
    eval_instance = evaluator(model=function_model)
    
    return [api_model, local_model, function_model]


if __name__ == "__main__":
    # Test the helper functions
    print("Testing custom model integration helpers...")
    
    # Test function wrapper
    def test_function(prompt):
        return f"Test response to: {prompt}"
    
    wrapper = create_function_wrapper(test_function)
    assert test_model_wrapper(wrapper), "Function wrapper test failed"
    print("Function wrapper test passed")
    
    # Test configuration
    config = create_proprietary_model_config("api", endpoint="https://test.com", api_key="test")
    assert "endpoint" in config, "Config creation failed"
    print("Configuration helper test passed")
    
    print("All custom model integration helpers working correctly!")
