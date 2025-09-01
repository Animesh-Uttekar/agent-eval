"""
AgentEval - The Effortless AI Agent Evaluation SDK

Single function call provides comprehensive evaluation with:
- Automatic test case generation (eliminates manual writing)
- Chain-of-Thought judges (15-30% accuracy improvement) 
- Domain-aware intelligence (healthcare, finance, legal)
- Local-first caching (10x performance improvement)
- Actionable insights and improvement recommendations

Usage:
    from agent_eval import evaluator
    
    # Zero configuration - auto-detects everything
    eval = evaluator(model=your_model)
    
    # Comprehensive evaluation with auto-generated tests
    result = eval.evaluate(
        prompt="Your prompt",
        model_output=your_agent("Your prompt")
    )
    
    # Get specific improvement recommendations  
    print(result.improvement_suggestions)
"""

from typing import Optional, Union, List, Dict, Any, Callable
from agent_eval.core.evaluator import Evaluator, EvaluationResult, EnhancedEvaluationResult
from agent_eval.domain.intelligence_engine import DomainType

# Import modular components for new API
from agent_eval.test_generation.generator import (
    ModularIntelligentTestGenerator,
    quick_analyze_agent,
    quick_generate_test_suite,
    build_custom_test_suite
)


def evaluator(
    model: Any = None,
    model_name: str = None, 
    provider: str = None,
    domain: Union[str, DomainType, None] = None,
    auto_generate_tests: bool = True,
    quality_gates: Optional[Dict[str, float]] = None,
    **kwargs
) -> Evaluator:
    """
    Create an evaluator with zero configuration.
    
    This single function replaces all setup complexity from other frameworks:
    - No YAML files needed (vs PromptFoo)
    - No complex configuration (vs LangSmith) 
    - No manual test writing (vs DeepEval)
    - No vendor lock-in (vs OpenAI Evals)
    
    Args:
        model: Your AI model/agent to evaluate
        model_name: Model name if needed
        provider: Provider name if needed
        domain: Domain context (healthcare, finance, legal, etc.)
        auto_generate_tests: Whether to auto-generate test scenarios
        quality_gates: Minimum quality thresholds
        **kwargs: Additional configuration
        
    Returns:
        Enhanced evaluator ready to use
    
    Examples:
        # Basic usage
        eval = evaluator(model=your_model)
        result = eval.evaluate(prompt="Explain photosynthesis", model_output=response)
        
        # Domain-specific evaluation
        eval = evaluator(model=medical_ai, domain="healthcare")
        result = eval.evaluate(prompt="What should I do about chest pain?", model_output=response)
        
        # Comprehensive agent evaluation  
        eval = evaluator(model=your_model, auto_generate_tests=True)
        results = eval.evaluate_agent(your_agent_function, num_test_scenarios=50)
    """
    
    return Evaluator(
        model=model,
        model_name=model_name,
        model_provider=provider,
        domain=domain,
        auto_generate_tests=auto_generate_tests,
        quality_gates=quality_gates,
        **kwargs
    )

IntelligentEvaluator = Evaluator

__all__ = [
    "evaluator",
    "Evaluator",
    "IntelligentEvaluator",
    "EvaluationResult",
    "EnhancedEvaluationResult",
    "ModularIntelligentTestGenerator",
    "quick_analyze_agent",
    "quick_generate_test_suite",
    "build_custom_test_suite",
    "DomainType",
]
