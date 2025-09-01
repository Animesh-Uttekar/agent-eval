"""
Main test suite orchestrator that coordinates all components using dependency injection.
This follows the Open/Closed Principle - new components can be added without modifying this class.
"""

from typing import List, Optional, Dict, Any, Callable
import logging
from dataclasses import dataclass

from .models import (
    TestScenario, AgentBehaviorProfile, ScenarioType, DifficultyLevel,
    TestSuiteGenerationRequest, TestSuiteGenerationResult
)
from .interfaces import (
    IAgentBehaviorAnalyzer, IScenarioGenerator, IDomainHandler, 
    IPromptGenerator, IMetricRecommender, IDistributionOptimizer
)
from .scenario_generators import ScenarioGeneratorFactory
from .behavior_analyzer import ModularAgentBehaviorAnalyzer
from .response_analyzer import ResponsePatternAnalyzer, CapabilityDetector, WeaknessDetector
from .prompt_generator import PromptGeneratorFactory


@dataclass
class TestSuiteConfig:
    """Configuration for test suite generation."""
    total_scenarios: int = 20
    scenario_distribution: Optional[Dict[ScenarioType, float]] = None
    difficulty_distribution: Optional[Dict[DifficultyLevel, float]] = None
    enable_behavior_analysis: bool = True
    enable_metric_recommendation: bool = True
    enable_distribution_optimization: bool = True


class ModularTestSuiteOrchestrator:
    """
    Main orchestrator that coordinates all test generation components.
    Uses dependency injection to maintain loose coupling and testability.
    """
    
    def __init__(
        self,
        behavior_analyzer: Optional[IAgentBehaviorAnalyzer] = None,
        domain_handler: Optional[IDomainHandler] = None,
        prompt_generator: Optional[IPromptGenerator] = None,
        metric_recommender: Optional[IMetricRecommender] = None,
        distribution_optimizer: Optional[IDistributionOptimizer] = None
    ):
        # Store dependencies first
        self.domain_handler = domain_handler
        self.prompt_generator = prompt_generator or PromptGeneratorFactory().get_generator("default")
        self.metric_recommender = metric_recommender
        self.distribution_optimizer = distribution_optimizer
        
        # Use dependency injection with defaults
        self.behavior_analyzer = behavior_analyzer or self._create_default_behavior_analyzer()
        
        # Create scenario generator factory
        self.scenario_factory = ScenarioGeneratorFactory(self.domain_handler)
        
        self.logger = logging.getLogger(__name__)
    
    def _create_default_behavior_analyzer(self) -> IAgentBehaviorAnalyzer:
        """Create default behavior analyzer with all components."""
        response_analyzer = ResponsePatternAnalyzer()
        capability_detector = CapabilityDetector()
        weakness_detector = WeaknessDetector()
        
        return ModularAgentBehaviorAnalyzer(
            response_analyzer=response_analyzer,
            capability_detector=capability_detector,
            weakness_detector=weakness_detector,
            metric_recommender=self.metric_recommender,
            distribution_optimizer=self.distribution_optimizer
        )
    
    def generate_test_suite(
        self,
        request: TestSuiteGenerationRequest,
        config: Optional[TestSuiteConfig] = None
    ) -> TestSuiteGenerationResult:
        """
        Generate a complete test suite based on the request and configuration.
        """
        config = config or TestSuiteConfig()
        
        try:
            # Step 1: Analyze agent behavior if enabled
            behavior_profile = None
            if config.enable_behavior_analysis and request.sample_responses:
                self.logger.info("Analyzing agent behavior...")
                behavior_profile = self.behavior_analyzer.analyze_agent_behavior(request)
            
            # Step 2: Determine scenario distribution
            distribution = self._determine_scenario_distribution(
                behavior_profile, config
            )
            
            # Step 3: Generate scenarios for each type
            all_scenarios = []
            for scenario_type, count in distribution.items():
                generator = self.scenario_factory.get_generator(scenario_type)
                scenarios = generator.generate_scenarios(behavior_profile, count)
                all_scenarios.extend(scenarios)
            
            # Step 4: Optimize scenarios if optimizer is available
            if self.distribution_optimizer and behavior_profile:
                all_scenarios = self.distribution_optimizer.optimize_scenario_distribution(
                    all_scenarios, behavior_profile
                )
            
            # Step 5: Get metric recommendations
            recommended_metrics = []
            if self.metric_recommender and behavior_profile:
                recommended_metrics = self.metric_recommender.recommend_metrics(behavior_profile)
            
            # Step 6: Create result
            result = TestSuiteGenerationResult(
                scenarios=all_scenarios,
                behavior_profile=behavior_profile,
                recommended_metrics=recommended_metrics,
                metadata={
                    "total_scenarios": len(all_scenarios),
                    "distribution": {str(k): v for k, v in distribution.items()},
                    "config": config.__dict__
                }
            )
            
            self.logger.info(f"Generated test suite with {len(all_scenarios)} scenarios")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating test suite: {e}")
            # Return minimal result on error
            return TestSuiteGenerationResult(
                scenarios=[],
                behavior_profile=None,
                recommended_metrics=[],
                metadata={"error": str(e)}
            )
    
    def _determine_scenario_distribution(
        self,
        behavior_profile: Optional[AgentBehaviorProfile],
        config: TestSuiteConfig
    ) -> Dict[ScenarioType, int]:
        """Determine how many scenarios of each type to generate."""
        
        # Use provided distribution or intelligent defaults
        if config.scenario_distribution:
            distribution = config.scenario_distribution
        elif behavior_profile:
            distribution = self._get_behavior_based_distribution(behavior_profile)
        else:
            distribution = self._get_default_distribution()
        
        # Convert percentages to counts
        scenario_counts = {}
        remaining_scenarios = config.total_scenarios
        
        for scenario_type, percentage in distribution.items():
            count = max(1, int(config.total_scenarios * percentage))
            scenario_counts[scenario_type] = count
            remaining_scenarios -= count
        
        # Distribute any remaining scenarios
        if remaining_scenarios > 0:
            # Add to normal scenarios by default
            scenario_counts[ScenarioType.NORMAL] = scenario_counts.get(
                ScenarioType.NORMAL, 0
            ) + remaining_scenarios
        
        return scenario_counts
    
    def _get_behavior_based_distribution(self, behavior_profile: AgentBehaviorProfile) -> Dict[ScenarioType, float]:
        """Get scenario distribution based on behavior profile."""
        distribution = {
            ScenarioType.NORMAL: 0.4,  # 40% normal scenarios
            ScenarioType.EDGE_CASE: 0.15,  # 15% edge cases
            ScenarioType.ADVERSARIAL: 0.1,  # 10% adversarial
            ScenarioType.PERFORMANCE: 0.15,  # 15% performance
            ScenarioType.CREATIVITY: 0.1,  # 10% creativity
            ScenarioType.DOMAIN_SPECIFIC: 0.1  # 10% domain-specific
        }
        
        # Adjust based on detected capabilities and weaknesses
        if behavior_profile.primary_capabilities:
            if "creativity" in behavior_profile.primary_capabilities:
                distribution[ScenarioType.CREATIVITY] += 0.1
                distribution[ScenarioType.NORMAL] -= 0.1
        
        if behavior_profile.weaknesses:
            # Increase edge case testing if weaknesses detected
            distribution[ScenarioType.EDGE_CASE] += 0.1
            distribution[ScenarioType.ADVERSARIAL] += 0.05
            distribution[ScenarioType.NORMAL] -= 0.15
        
        if behavior_profile.domain:
            # Increase domain-specific testing if domain is detected
            distribution[ScenarioType.DOMAIN_SPECIFIC] += 0.1
            distribution[ScenarioType.NORMAL] -= 0.1
        
        return distribution
    
    def _get_default_distribution(self) -> Dict[ScenarioType, float]:
        """Get default scenario distribution when no behavior profile is available."""
        return {
            ScenarioType.NORMAL: 0.5,
            ScenarioType.EDGE_CASE: 0.2,
            ScenarioType.ADVERSARIAL: 0.1,
            ScenarioType.PERFORMANCE: 0.1,
            ScenarioType.CREATIVITY: 0.05,
            ScenarioType.DOMAIN_SPECIFIC: 0.05
        }
    
    def analyze_agent_only(self, request: TestSuiteGenerationRequest) -> Optional[AgentBehaviorProfile]:
        """Analyze agent behavior without generating test scenarios."""
        if not request.sample_responses:
            return None
        
        return self.behavior_analyzer.analyze_agent_behavior(request)
    
    def generate_scenarios_for_profile(
        self,
        behavior_profile: AgentBehaviorProfile,
        scenario_type: ScenarioType,
        count: int = 5
    ) -> List[TestScenario]:
        """Generate specific scenarios for a given behavior profile and type."""
        generator = self.scenario_factory.get_generator(scenario_type)
        return generator.generate_scenarios(behavior_profile, count)
    
    def update_behavior_analyzer(self, behavior_analyzer: IAgentBehaviorAnalyzer):
        """Update the behavior analyzer (useful for testing or configuration changes)."""
        self.behavior_analyzer = behavior_analyzer
    
    def update_domain_handler(self, domain_handler: IDomainHandler):
        """Update the domain handler and recreate scenario factory."""
        self.domain_handler = domain_handler
        self.scenario_factory = ScenarioGeneratorFactory(domain_handler)
        
        # Update behavior analyzer if it supports domain handler
        if hasattr(self.behavior_analyzer, 'domain_handler'):
            self.behavior_analyzer.domain_handler = domain_handler


class TestSuiteBuilder:
    """
    Builder pattern implementation for creating test suites with fluent interface.
    Provides an easy-to-use API for configuring and generating test suites.
    """
    
    def __init__(self):
        self.orchestrator = ModularTestSuiteOrchestrator()
        self.config = TestSuiteConfig()
        self.request = None
    
    def with_sample_responses(self, sample_responses: List[str]) -> 'TestSuiteBuilder':
        """Add sample responses for behavior analysis."""
        self.request = TestSuiteGenerationRequest(sample_responses=sample_responses)
        return self
    
    def with_total_scenarios(self, count: int) -> 'TestSuiteBuilder':
        """Set the total number of scenarios to generate."""
        self.config.total_scenarios = count
        return self
    
    def with_scenario_distribution(self, distribution: Dict[ScenarioType, float]) -> 'TestSuiteBuilder':
        """Set custom scenario type distribution."""
        self.config.scenario_distribution = distribution
        return self
    
    def with_domain_handler(self, domain_handler: IDomainHandler) -> 'TestSuiteBuilder':
        """Add domain-specific handling."""
        self.orchestrator.update_domain_handler(domain_handler)
        return self
    
    def with_behavior_analyzer(self, behavior_analyzer: IAgentBehaviorAnalyzer) -> 'TestSuiteBuilder':
        """Use custom behavior analyzer."""
        self.orchestrator.update_behavior_analyzer(behavior_analyzer)
        return self
    
    def enable_behavior_analysis(self, enabled: bool = True) -> 'TestSuiteBuilder':
        """Enable or disable behavior analysis."""
        self.config.enable_behavior_analysis = enabled
        return self
    
    def enable_metric_recommendation(self, enabled: bool = True) -> 'TestSuiteBuilder':
        """Enable or disable metric recommendation."""
        self.config.enable_metric_recommendation = enabled
        return self
    
    def build(self) -> TestSuiteGenerationResult:
        """Build and generate the test suite."""
        if not self.request:
            # Create empty request if none provided
            self.request = TestSuiteGenerationRequest(sample_responses=[])
        
        return self.orchestrator.generate_test_suite(self.request, self.config)
