"""
Core module for modular test generation following SOLID principles.

This module provides a clean, modular architecture for intelligent test generation
with the following key components:

- Models: Core data structures and enums
- Interfaces: Abstract base classes following Interface Segregation Principle
- Analyzers: Behavior and response analysis components
- Generators: Scenario and prompt generation with Strategy pattern
- Orchestrator: Main coordinator using dependency injection

Example usage:
    from agent_eval.test_generation.core import TestSuiteBuilder, ScenarioType
    
    # Simple usage with builder pattern
    result = (TestSuiteBuilder()
              .with_sample_responses(["Hello", "How are you?"])
              .with_total_scenarios(10)
              .enable_behavior_analysis()
              .build())
    
    # Advanced usage with custom components
    orchestrator = ModularTestSuiteOrchestrator(
        behavior_analyzer=custom_analyzer,
        domain_handler=custom_domain_handler
    )
"""

# Core data structures
from .models import (
    TestScenario,
    AgentBehaviorProfile,
    ScenarioType,
    DifficultyLevel,
    ResponsePattern,
    TestSuiteGenerationRequest,
    TestSuiteGenerationResult
)

# Interfaces for dependency injection
from .interfaces import (
    IAgentBehaviorAnalyzer,
    IResponsePatternAnalyzer,
    ICapabilityDetector,
    IWeaknessDetector,
    IScenarioGenerator,
    IDomainHandler,
    IPromptGenerator,
    IMetricRecommender,
    IDistributionOptimizer
)

# Analysis components
from .response_analyzer import (
    ResponsePatternAnalyzer,
    CapabilityDetector,
    WeaknessDetector
)

from .behavior_analyzer import ModularAgentBehaviorAnalyzer

# Generation components
from .prompt_generator import (
    PromptGeneratorFactory,
    DefaultPromptGenerator,
    CreativePromptGenerator,
    TechnicalPromptGenerator
)

from .scenario_generators import (
    ScenarioGeneratorFactory,
    NormalScenarioGenerator,
    EdgeCaseScenarioGenerator,
    AdversarialScenarioGenerator,
    PerformanceScenarioGenerator,
    CreativityScenarioGenerator,
    DomainSpecificScenarioGenerator
)

# Main orchestrator and builder
from .orchestrator import (
    ModularTestSuiteOrchestrator,
    TestSuiteBuilder,
    TestSuiteConfig
)

# Convenience exports for easy usage
__all__ = [
    # Data structures
    "TestScenario",
    "AgentBehaviorProfile", 
    "ScenarioType",
    "DifficultyLevel",
    "ResponsePattern",
    "TestSuiteGenerationRequest",
    "TestSuiteGenerationResult",
    
    # Interfaces
    "IAgentBehaviorAnalyzer",
    "IResponsePatternAnalyzer", 
    "ICapabilityDetector",
    "IWeaknessDetector",
    "IScenarioGenerator",
    "IDomainHandler",
    "IPromptGenerator",
    "IMetricRecommender",
    "IDistributionOptimizer",
    
    # Implementation classes
    "ResponsePatternAnalyzer",
    "CapabilityDetector", 
    "WeaknessDetector",
    "ModularAgentBehaviorAnalyzer",
    "PromptGeneratorFactory",
    "ScenarioGeneratorFactory",
    
    # Individual generators (for advanced usage)
    "NormalScenarioGenerator",
    "EdgeCaseScenarioGenerator", 
    "AdversarialScenarioGenerator",
    "PerformanceScenarioGenerator",
    "CreativityScenarioGenerator",
    "DomainSpecificScenarioGenerator",
    
    # Main API
    "ModularTestSuiteOrchestrator",
    "TestSuiteBuilder",
    "TestSuiteConfig"
]
