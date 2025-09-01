"""
Modular test generation API that uses the new architecture.
This provides a clean interface for users who want to use the modular components.
"""

from typing import List, Optional, Dict, Any, Union
import logging
from pathlib import Path

from .core import (
    TestSuiteBuilder, ModularTestSuiteOrchestrator, TestSuiteConfig,
    TestScenario, AgentBehaviorProfile, ScenarioType, DifficultyLevel,
    TestSuiteGenerationRequest, TestSuiteGenerationResult,
    IAgentBehaviorAnalyzer, IDomainHandler, IPromptGenerator,
    IMetricRecommender, IDistributionOptimizer
)


class ModularIntelligentTestGenerator:
    """
    Modern, modular version of the intelligent test generator.
    Uses dependency injection and follows SOLID principles for maintainability.
    """
    
    def __init__(
        self,
        behavior_analyzer: Optional[IAgentBehaviorAnalyzer] = None,
        domain_handler: Optional[IDomainHandler] = None,
        prompt_generator: Optional[IPromptGenerator] = None,
        metric_recommender: Optional[IMetricRecommender] = None,
        distribution_optimizer: Optional[IDistributionOptimizer] = None
    ):
        """
        Initialize the modular test generator with optional custom components.
        
        Args:
            behavior_analyzer: Custom behavior analyzer implementation
            domain_handler: Domain-specific logic handler
            prompt_generator: Custom prompt generation strategy
            metric_recommender: Custom metric recommendation engine
            distribution_optimizer: Custom scenario distribution optimizer
        """
        self.orchestrator = ModularTestSuiteOrchestrator(
            behavior_analyzer=behavior_analyzer,
            domain_handler=domain_handler,
            prompt_generator=prompt_generator,
            metric_recommender=metric_recommender,
            distribution_optimizer=distribution_optimizer
        )
        
        self.logger = logging.getLogger(__name__)
    
    def generate_test_suite(
        self,
        sample_responses: List[str],
        total_scenarios: int = 20,
        scenario_distribution: Optional[Dict[ScenarioType, float]] = None,
        enable_behavior_analysis: bool = True,
        enable_metric_recommendation: bool = True,
        **kwargs
    ) -> TestSuiteGenerationResult:
        """
        Generate a comprehensive test suite based on sample agent responses.
        
        Args:
            sample_responses: List of sample responses from the agent to analyze
            total_scenarios: Total number of test scenarios to generate
            scenario_distribution: Custom distribution of scenario types (as percentages)
            enable_behavior_analysis: Whether to analyze agent behavior first
            enable_metric_recommendation: Whether to recommend evaluation metrics
            **kwargs: Additional configuration options
        
        Returns:
            TestSuiteGenerationResult with scenarios, behavior profile, and recommendations
        """
        config = TestSuiteConfig(
            total_scenarios=total_scenarios,
            scenario_distribution=scenario_distribution,
            enable_behavior_analysis=enable_behavior_analysis,
            enable_metric_recommendation=enable_metric_recommendation,
            **kwargs
        )
        
        request = TestSuiteGenerationRequest(
            sample_responses=sample_responses,
            metadata=kwargs
        )
        
        return self.orchestrator.generate_test_suite(request, config)
    
    def analyze_agent_behavior(self, sample_responses: List[str]) -> Optional[AgentBehaviorProfile]:
        """
        Analyze agent behavior without generating test scenarios.
        
        Args:
            sample_responses: Sample responses to analyze
            
        Returns:
            AgentBehaviorProfile or None if analysis fails
        """
        request = TestSuiteGenerationRequest(sample_responses=sample_responses)
        return self.orchestrator.analyze_agent_only(request)
    
    def generate_scenarios_by_type(
        self,
        behavior_profile: AgentBehaviorProfile,
        scenario_type: ScenarioType,
        count: int = 5
    ) -> List[TestScenario]:
        """
        Generate specific scenarios of a given type.
        
        Args:
            behavior_profile: Analyzed behavior profile
            scenario_type: Type of scenarios to generate
            count: Number of scenarios to generate
            
        Returns:
            List of generated test scenarios
        """
        return self.orchestrator.generate_scenarios_for_profile(
            behavior_profile, scenario_type, count
        )
    
    def quick_test_suite(
        self,
        sample_responses: List[str],
        focus_area: str = "balanced"
    ) -> TestSuiteGenerationResult:
        """
        Generate a quick test suite with predefined configurations.
        
        Args:
            sample_responses: Sample responses to analyze
            focus_area: Area to focus on ("balanced", "robustness", "creativity", "performance")
        
        Returns:
            TestSuiteGenerationResult
        """
        distributions = {
            "balanced": {
                ScenarioType.NORMAL: 0.4,
                ScenarioType.EDGE_CASE: 0.2,
                ScenarioType.ADVERSARIAL: 0.15,
                ScenarioType.PERFORMANCE: 0.15,
                ScenarioType.CREATIVITY: 0.1
            },
            "robustness": {
                ScenarioType.NORMAL: 0.2,
                ScenarioType.EDGE_CASE: 0.4,
                ScenarioType.ADVERSARIAL: 0.3,
                ScenarioType.PERFORMANCE: 0.1
            },
            "creativity": {
                ScenarioType.NORMAL: 0.3,
                ScenarioType.CREATIVITY: 0.4,
                ScenarioType.EDGE_CASE: 0.2,
                ScenarioType.ADVERSARIAL: 0.1
            },
            "performance": {
                ScenarioType.NORMAL: 0.2,
                ScenarioType.PERFORMANCE: 0.5,
                ScenarioType.EDGE_CASE: 0.2,
                ScenarioType.ADVERSARIAL: 0.1
            }
        }
        
        distribution = distributions.get(focus_area, distributions["balanced"])
        
        return self.generate_test_suite(
            sample_responses=sample_responses,
            total_scenarios=15,
            scenario_distribution=distribution
        )


# Convenience functions for easy usage
def quick_analyze_agent(sample_responses: List[str]) -> Optional[AgentBehaviorProfile]:
    """Quick analysis of agent behavior."""
    generator = ModularIntelligentTestGenerator()
    return generator.analyze_agent_behavior(sample_responses)


def quick_generate_test_suite(
    sample_responses: List[str],
    total_scenarios: int = 20,
    focus_area: str = "balanced"
) -> TestSuiteGenerationResult:
    """Quick test suite generation with sensible defaults."""
    generator = ModularIntelligentTestGenerator()
    return generator.quick_test_suite(sample_responses, focus_area)


def build_custom_test_suite() -> TestSuiteBuilder:
    """Create a test suite builder for custom configuration."""
    return TestSuiteBuilder()


# Example usage
if __name__ == "__main__":
    # Example 1: Quick usage
    sample_responses = [
        "Hello! I'm an AI assistant. How can I help you today?",
        "I can help with analysis, writing, coding, and problem-solving tasks.",
        "Let me break down this complex problem into smaller parts..."
    ]
    
    result = quick_generate_test_suite(sample_responses, focus_area="balanced")
    print(f"Generated {len(result.scenarios)} scenarios")
    
    # Example 2: Builder pattern
    result = (build_custom_test_suite()
              .with_sample_responses(sample_responses)
              .with_total_scenarios(25)
              .enable_behavior_analysis()
              .enable_metric_recommendation()
              .build())
    
    # Example 3: Direct orchestrator usage for advanced control
    orchestrator = ModularTestSuiteOrchestrator()
    config = TestSuiteConfig(total_scenarios=30)
    request = TestSuiteGenerationRequest(sample_responses=sample_responses)
    result = orchestrator.generate_test_suite(request, config)
